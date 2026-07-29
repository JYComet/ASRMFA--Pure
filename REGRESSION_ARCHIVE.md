# 异常存档库 / Regression Archive

用于代码修改时的回归校对。每项记录一个已修复的逻辑冲突场景，
修改相关代码时需验证该场景不被复现。

---

## 索引

| # | 日期 | 文件 | 标题 |
|---|------|------|------|
| 1 | 2026-07-17 | postprocess_textgrids.py | MFA尾静音被snap回词而非合并到标点 (jie2) |
| 2 | 2026-07-17 | postprocess_textgrids.py | Phase 1 静音合并 vs Phase 3 Rule 3 顺位冲突 (er4) |
| 3 | 2026-07-17 | postprocess_textgrids.py | 跨词界 eps 被 xmax 上限检查漏掉 (le5) |
| 4 | 2026-07-17 | postprocess_textgrids.py | 短NVV强制CTC导致前词被裁短 (ti2/BREATHING) |
| 5 | 2026-07-17 | postprocess_textgrids.py | 词首前拉：能量谷底检测 (na2) |
| 6 | 2026-07-17 | postprocess_textgrids.py | 静音段延伸+end-trimming回截冲突 (ji2) |
| 7 | 2026-07-23 | postprocess_textgrids.py | pinyin_phones首音素与词界间隙 (ru2→r, NVV邻接) |
| 8 | 2026-07-27 | postprocess_textgrids.py | 标点间隙中<sp0>因has_punct跳过合并→mid_sp误报 |
| 9 | 2026-07-27 | postprocess_textgrids.py | 标点后<spN>未被标点吸收→mid_sp误报 |
| 10 | 2026-07-27 | postprocess_textgrids.py | Hanzi tier拼音残留：NW对齐+过度宽松匹配→CJK被跳过 (qie4/切) |
| 11 | 2026-07-27 | postprocess_textgrids.py | cjk_mismatch/hanzi_pinyin 检查在路径决策之后执行→文件未被重定向 |
| 12 | 2026-07-27 | postprocess_textgrids.py | CTC snap重叠修复产生倒置interval + 尾静音未被最后标点吸收 (xia4/ta1) |
| 13 | 2026-07-27 | postprocess_textgrids.py, finalize_textgrids.py | NVV 大小写不一致：MFA 小写 laughter 不被识别/包裹 → pinyin_phones 残留小写 |
| 14 | 2026-07-28 | postprocess_textgrids.py | Hanzi tier NVV 括号污染：<_build_hanzi_tier alpha 分支未 strip <> → label 残留 `<SURPRISE-WA>` |
| 15 | 2026-07-28 | postprocess_textgrids.py | _extract_word_chars 拆分 NVV 尖括号：<LAUGHTER> → [<, LAUGHTER, >] |
| 16 | 2026-07-28 | run_pipeline.py | MFA --fine_tune 默认开启→边界漂移+对齐失败 |
| 17 | 2026-07-28 | postprocess_textgrids.py, ctc_prealign.py | NVV改写+首词标点+音素断层+轨道不同步+连字符替换 |
| 18 | 2026-07-29 | ctc_prealign.py | NVASR 情绪标签后残留开头标点 → hanzi 首词为标点 |
| 19 | 2026-07-29 | postprocess_textgrids.py | _snap_to_ctc 丢弃开头静音 → hanzi 缺失 <sp1> interval |
| 20 | 2026-07-29 | postprocess_textgrids.py | strip_edge_punctuation 尾随标点过度剥离 → 句尾 。！？ 全部丢失 |
| 21 | 2026-07-29 | postprocess_textgrids.py, pipeline_utils.py | is_punct 误判 <spN> 为标点 → strip_edge_punctuation 开头静音被吸收进首词 |
| 22 | 2026-07-29 | postprocess_textgrids.py | MFA 对齐偏差→snap 修正后标点被 <spN> 替换→被吞标点恢复 |
| 23 | 2026-07-29 | run_pipeline.py | step_normalize_punct 缺少 NVV 连字符保护 → QUESTION-EI 被拆成 QUESTION，EI |
| 24 | 2026-07-29 | ctc_prealign.py | ASR 空输出先写 TextGrid 再检测 → 留下孤立的空 TextGrid |

---

## 项目管线结构 / Pipeline Architecture

### 入口

```
scripts/run_pipeline.py          — 主控脚本，调度全部步骤
scripts/streaming_pipeline.py    — 流式批处理（NAS→本地SSD→回传）
```

### 三种管线模式

| 模式 | 步骤序列 | 适用场景 |
|------|---------|---------|
| `full` | trim → resample → prealign → normalize_punct → normalize → normalize_ria → normalize_en → adjust → align → align_en → postprocess | 原始 WAV 无任何预处理 |
| `nvrasr_fallback` | prealign → pad_silence → normalize_punct → normalize → normalize_ria → normalize_en → resample → adjust → align → align_en → postprocess | 音频已预裁剪，只需重新跑 NVASR |
| `ctc_ready` | link → pad_silence → normalize_punct → normalize → normalize_ria → normalize_en → resample → adjust → align → align_en → postprocess | 已有 NVASR CTC 输出，跳过 trim + prealign |

---

### 步骤详解（full 模式执行顺序）

#### 1. trim — 音频预处理
- **脚本**: `scripts/trim_silence_batch.py`
- **功能**: 切除句内长静音（>1.0s），规范首尾静音至 0.5s
- **输入**: 原始 WAV
- **输出**: `workspace/audio/`

#### 2. resample — 重采样至 16kHz
- **代码**: `run_pipeline.py::step_resample_for_mfa`
- **功能**: 多线程重采样至 16kHz 单声道（MFA 要求）
- **输入**: `audio/`
- **输出**: `workspace/audio_16k/`

#### 3. prealign — NVASR CTC 预对齐
- **脚本**: `scripts/ctc_prealign.py`
- **功能**: NVASR（SenseVoice-Small）CTC 强制对齐，产出词级锚点
- **输入**: `audio_16k/` + 可选参考文本
- **输出**: `workspace/ctc_pretg/`

  **内部处理流程**:
  ```
  NVASR Encoder → CTC logits
    ├─ blank-frame NVV bias（提升 NVV token 检测）
    ├─ pause→ellipsis 注入（空白帧 ≥ 阈值 → 插入 … token）
    ├─ NVV bracket 转换（[Surprise-oh] → SURPRISE-OH）
    ├─ ria 变体还原（rui4 ya4 → ria）
    ├─ 数字→中文（cn2an）
    ├─ 标点规范化（normalize_punct_inline）
    └─ Token 级 CTC 时间戳解码
         │
         ├─ .lab              MFA 语料（pinyin+NVV）
         ├─ .TextGrid         CTC 锚点（words tier）
         ├─ _tokens.jsonl     逐词时间戳
         ├─ _punct.json       标点时间戳
         ├─ _text_cn.txt      中文文本（raw_text tier 用）
         └─ _text_raw.txt     ASR 原始输出（含 NVV 标签）
  ```

  **输出后处理**（`main()` 末尾自动调用）:
  - `_normalize_punct`: ASCII→CJK 标点映射 + 相邻合并 + 非白名单替换。**含 NVV 连字符保护（Case 17-E）**
  - `_normalize_numerals`: 阿拉伯数字→中文（cn2an）
  - `_normalize_english`: 英文 token 规范化（`scripts/normalize_english_tokens.py`）
  - `_normalize_ria`: ria 音译还原（仅 ctc_ready/nvrasr_fallback 模式）

  **关键函数**:
  | 函数 | 作用 |
  |------|------|
  | `make_patched_inference` | 批量推理，返回 CTC 对齐结果 |
  | `chars_and_pinyin` | 汉字→拼音映射（pypinyin） |
  | `nvv_to_mfa` | `[Surprise-oh]` → `SURPRISE-OH` |
  | `write_textgrid` | 写 CTC 锚点 TextGrid |
  | `clean_unsupported_punct` | 过滤白名单外标点字符 |
  | `_normalize_punct` | 标点规范化（含 NVV 连字符保护） |
  | `_normalize_numerals` | 数字→中文 |

#### 4. normalize_punct — 标点规范化
- **代码**: `run_pipeline.py::step_normalize_punct`
- **功能**: ASCII 标点→CJK，合并相邻标点，同步更新 `_punct.json`
- **注意**: 若 prealign 已运行输出后处理，此步骤操作的是已有文件

#### 5. normalize — 数字规范化
- **代码**: `run_pipeline.py::step_normalize_text`
- **功能**: 阿拉伯数字→中文（cn2an），更新 `_text_cn.txt` + `.lab`

#### 6. normalize_ria — ria 音译还原
- **代码**: `run_pipeline.py::step_normalize_ria`
- **功能**: `rui4 ya4` → `ria`，更新 `.lab` + `_tokens.jsonl`

#### 7. normalize_en — 英文 token 规范化
- **脚本**: `scripts/normalize_english_tokens.py`
- **功能**: NVASR 拼音碎片→英文原词（如 `li ve` → `live`）

#### 8. adjust — CTC 边界能量修正
- **脚本**: `scripts/adjust_ctc_boundaries.py`
- **功能**: 音频能量分析修正 CTC 锚点边界（词首能量上升→前推，词尾能量下降→后延）
- **输入**: `ctc_pretg/` + `audio_16k/`
- **输出**: `workspace/ctc_pretg_adj/`
- **保护**: NVV token 不被修改；标点同步调整

  **关键函数**:
  | 函数 | 作用 |
  |------|------|
  | `adjust_boundaries` | 主逻辑：能量上升/下降检测 + 边界调整 |
  | `rebuild_textgrid` | 从调整后的 tokens 重建 TextGrid |
  | `process_one` | 单文件处理入口 |

#### 9. validate — MFA 语料验证
- **代码**: `run_pipeline.py::step_mfa_validate`
- **默认跳过**（`skip_validate: true`），MFA align 内部已验证

#### 10. align — MFA 强制对齐
- **代码**: `run_pipeline.py::step_mfa_align`
- **功能**: MFA 使用 NVASR `.lab` 作语料 + CTC TextGrid 作锚点 → 音素级对齐
- **输入**: `ctc_pretg_adj/` + `audio_16k/` + `mfa_ipa.dict`
- **输出**: `workspace/aligned/`（含 words + phones 两个 tier）
- **关键参数**:
  - `--beam 20 --retry_beam 80`：Viterbi 波束宽度
  - `--fine_tune false`：默认关闭（Case 16），CTC 锚点已由 adjust 修正
  - `--fine_tune_boundary_tolerance 0.02`：仅 fine_tune=true 时生效

#### 11. align_en — 英文 MFA 对齐
- **脚本**: `scripts/align_english_mfa.py`
- **功能**: 从 CTC 边界提取英文词段 → English MFA（`english_us_arpa`）对齐
- **输入**: `ctc_pretg_adj/` + `audio_16k/` + `cmudict.dict`
- **输出**: `workspace/en_phones/`（`*_en_phones.json`）

  **关键函数**:
  | 函数 | 作用 |
  |------|------|
  | `find_english_segments` | 扫描 CTC 输出，识别英文词段 |
  | `build_en_corpus` | 构建 English MFA 语料目录 |
  | `build_en_dict` | 构建英文发音词典（CMUdict + G2P fallback） |
  | `run_en_mfa` | 调用 English MFA align |
  | `collect_en_phones` | 收集 English MFA 音素→写入 JSON |

#### 12. postprocess — 后处理（最复杂步骤）
- **脚本**: `scripts/postprocess_textgrids.py`
- **功能**: 从 MFA 对齐结果构建最终 5-tier TextGrid + 质检
- **输入**: `aligned/` + `ctc_pretg_adj/` + `en_phones/` + dicts
- **输出**: `workspace/output/`（通过质检）+ `workspace/filtered/`（未通过）

  **Phase 执行顺序**:
  ```
  Phase 1 — 声学前处理
    ├─ merge_short_silences    短静音合并
    ├─ fix_short_words          短词修复（能量检测）
    └─ 重建 pinyin_phones       （若 merge/fix 修改了边界）

  Phase 2 — 文本修正 + tier 定型
    ├─ silence relabel          <eps>/sil → <spN> 按duration
    ├─ _finalise_textgrid       构建 raw_text/pinyin/hanzi/words/pp
    │   ├─ normalize NVV brackets → <BREATHING>
    │   └─ sp1 normalization
    ├─ 标点-静音交叉校验
    └─ 输出路径选择（output vs filtered 预判）

  Phase 3 — 边界调整（顺序不可变）
    ├─ A. _snap_to_ctc           MFA边界→CTC锚点（权威）
    ├─ B. _refine_boundaries_by_energy  能量微调
    │   └─ 同步 pinyin_phones    ← 修改 words 后立即重建 pp
    └─ C. _inject_punctuation    标点注入 words tier

  Phase 3.5 — 英文音素注入
    └─ _apply_en_phones          英文 MFA 音素→phones tier
        └─ 同步 pinyin_phones    ← 修改 phones 后立即重建 pp

  Phase 4 — 后边界处理（顺序不可变）
    ├─ D.  handle_unexpected_silences  意外静音检测
    ├─ D2. absorb_nvv_trailing         NVV 吞并尾部标点+静音链
    ├─ D3. absorb_silence_into_punct   残余静音→标点吸收
    ├─ D4. strip_edge_punctuation      边缘标点剥离（Case 17-B）
    ├─ ── SYNC ──                      _sync_derived_tiers
    │   └─ 从 words+phones 重建 hanzi + pinyin_phones（Case 17-F）
    ├─ E.  NVV+ellipsis 无条件合并
    ├─ F.  _merge_nvv_ellipsis        能量判断 NVV+省略号合并
    ├─ G.  _extend_word_into_ellipsis  能量判断词延伸入省略号
    └─ ── SYNC ──                      _sync_derived_tiers

  Phase 5 — 最终文本同步 + 质检
    ├─ NVV 边界回退到 CTC 锚点
    ├─ 被吞标点检测 + raw_text 同步删除
    ├─ _normalize_word_spellings      英文词碎片→规范拼写
    │   └─ NVV token 文本保护（Case 17-A）
    ├─ _build_hanzi_tier              从 words + raw_text 重建 hanzi
    ├─ build_pinyin_phones_tier       从 phones + words 重建 pinyin_phones
    │   ├─ 泄漏音素过滤（Case 17-C）
    │   ├─ en_mfa_windows 缺失时 fallback（Case 17-D）
    │   └─ NVV token 自引用音素
    └─ QC 检查（10+ 过滤规则）→ output/ 或 filtered/
  ```

  **Tier 同步架构**（Case 17-F 建立）:
  ```
  words tier  ← 唯一权威数据源
     │
     ├─ hanzi tier       ← 从 words + raw_text 派生
     └─ pinyin_phones    ← 从 phones + words + pinyin_dict 派生
  
  规则: 任何修改 words tier 边界/文本的操作，必须立即调用
        _sync_derived_tiers() 重建 hanzi + pinyin_phones。
        不允许延迟到 Phase 5。"修改谁，同步全部"。
  ```

  **关键函数**:
  | 函数 | Phase | 作用 |
  |------|-------|------|
  | `merge_short_silences` | 1 | 合并短静音段 |
  | `fix_short_words` | 1 | 能量检测修复短词边界 |
  | `_finalise_textgrid` | 2 | 构建 5-tier TextGrid |
  | `_snap_to_ctc` | 3A | MFA→CTC 边界权威对齐 |
  | `_refine_boundaries_by_energy` | 3B | 能量微调词边界 |
  | `_inject_punctuation` | 3C | CTC 标点注入 words tier |
  | `_apply_en_phones` | 3.5 | 英文 MFA 音素注入 phones tier |
  | `absorb_nvv_trailing` | 4-D2 | NVV 吞并尾部标点+静音 |
  | `absorb_silence_into_punct` | 4-D3 | 残余静音→标点吸收 |
  | `strip_edge_punctuation` | 4-D4 | 边缘标点剥离 |
  | `_sync_derived_tiers` | 4-sync | **统一同步：words→hanzi+pp** |
  | `_merge_nvv_ellipsis` | 4-F | 能量判断 NVV+省略号合并 |
  | `_extend_word_into_ellipsis` | 4-G | 能量判断词延伸入省略号 |
  | `_normalize_word_spellings` | 5 | 英文碎片→规范拼写（含 NVV 保护） |
  | `_build_hanzi_tier` | 5 | 从 words 重建 hanzi |
  | `build_pinyin_phones_tier` | 5 | 从 phones+words 重建 pinyin_phones |

---

## 修改点汇总

| ID | 位置 | 修改 |
|----|------|------|
| A | `_snap_to_ctc` ~2519 | Rule 3 绕过: ratio_skip (pattern a+b) |
| B | `_snap_to_ctc` ~2569 | 中间点保护: keep_mfa_end |
| C | `_inject_punctuation` ~1877 | gap kind 扩展: "gap" 可被标点吸收 |
| D | `_snap_to_ctc` ~2541 | Pattern (b) 缩进修复 |
| E | `_snap_to_ctc` ~2598 | 重叠防护: prev_was_silence_extended |
| F | 已移除 | gap_was_merged 被能量分析否决 |
| G | `_snap_to_ctc` ~2526,~2572 | has_trailing_sil: xmax→xmin |
| H | `_snap_to_ctc` ~2484 | NVV 短时长例外 (<100ms 用MFA) |
| I | `_refine_boundaries_by_energy` ~2320 | 词尾能量延伸 + NVV前向延伸 |
| J | `_refine_boundaries_by_energy` ~2320 | 词首前拉：能量谷底检测 |
| K | `_refine_boundaries_by_energy` ~2415,~2623 | 静音段延伸(K1-K3)+punct全范围检查(K5)+延伸保护(K4) |
| Q | `build_pinyin_phones_tier` ~533,~539 | 逻辑修复: 首音素start/末音素end 无条件 snap 到词界 |
| R | `process_one` ~3708 | Phase 3.B 后同步：能量调整 words 后立即重建 pinyin_phones |
| S | `process_one` ~3760 | Phase 3.5 后同步：英语音素注入后立即重建 pinyin_phones |
| T | `_snap_to_ctc` ~3118 | words tier 连续性清理：吸收 ≤5ms 词间间隙 |
| U | `handle_unexpected_silences` ~636-641 | `<sp0>` 无条件合并：`has_punct` 不再拦截 `<sp0>` |
| V | `handle_unexpected_silences` ~668-693 | has_punct 时 `<sp0>` 合并到标点而非前词 |
| W | `step_mfa_align` ~1002, DEFAULT_CFG ~205 | **Case 16**: fine_tune 默认 True→False，tolerance 0.1→0.02 |
| X | `_normalize_word_spellings` ~1374 | **Case 17-A**: NVV token 文本永不改写 |
| Y | `strip_edge_punctuation` ~879 (新增) | **Case 17-B**: 边缘标点剥离函数 |
| Z | `build_pinyin_phones_tier` ~469 | **Case 17-C**: 泄漏音素检测（首音素起点>词起点30%→清空） |
| AA | `build_pinyin_phones_tier` ~521 | **Case 17-D**: en_mfa_windows 缺失时清空 word_phones |
| AB | `_normalize_punct` ~651 (ctc_prealign.py) | **Case 17-E**: 字母间 `-` 不当作标点（NVV 连字符保护） |
| AC | `_sync_derived_tiers` ~879 (新增) | **Case 17-F**: 统一 tier 同步函数 + Phase 4 调用点 |
| W1 | `absorb_nvv_trailing` ~758-851 (新增) | Pass 1: NVV 吸收标点+静音链 (D2 步骤) |
| W2 | `absorb_silence_into_punct` ~854-920 (新增) | Pass 2 (兜底): 标点吸收残余 `<spN>` (D3 步骤) |
| X | `_word_matches` ~1069-1076 | 拼音→英文匹配增加元音约束 (≥2 vowels) |
| Y | `_build_hanzi_tier` ~1173-1298 (重写) | 顺序CJK映射替代NW全局对齐 + 贪心alpha匹配 |
| Z | `_alpha_text_matches` ~1136-1170 (新增) | Alpha token ↔ ref unit 匹配（从旧 _word_matches 抽取） |
| AA1 | `process_one` ~4717-4749 | Hanzi tier 完整性检查移至路径决策之前（修复时序bug） |
| AA2 | `process_one` ~4724-4734 (新增) | 直接 pinyin 残留扫描：`is_pinyin_syllable` on hanzi labels → `hanzi_pinyin` |
| AC | `_snap_to_ctc` ~3317 | 重叠修复后防倒置 interval：`word_end < word_start` 时延伸 word_end |
| AD | `_inject_punctuation` ~2277 | 最后标点保留条件：`m[0] < punct_start` → `m[1] <= punct_start + 0.001` |
| AE | `_NVV_PATTERN` ~87-91 | 添加 `re.IGNORECASE` + 字符类 `[A-Za-z-]` 支持小写 NVV 匹配 |
| AF | `_finalize_textgrid` ~141 | NVV 替换统一大写：`r"<\1>"` → `lambda m: f"<{m.group(1).upper()}>"` |
| AG | `build_pinyin_phones_tier` ~485-488 | NVV 文本规范化：`f"<{...strip('<>').upper()}>"` 防大小写+括号残留 |
| AH | `finalize_textgrids.py` ~50-54 | NVV 规范化：strip+upper 防双重包裹和大小写不一致 |
| AI | `_build_hanzi_tier` ~1246-1278 | Alpha 分支 strip `<>`：`clean_token` 参与匹配和 fallback label，防 hanzi 污染 |
| AJ | `_extract_word_chars` ~1004-1032 | `<` `>` 特殊处理：flush + 开新 group / flush + 闭合，防 NVV 被拆分 |
| AK | `strip_edge_punctuation` ~997-1017 | **Case 20**: 删除尾随标点剥离逻辑（镜像设计错误——开头和尾随不是对称场景） |
| AL | `_inject_punctuation` ~2482-2484 | **Case 20-B**: 最后标点 30ms 地板，防末词 xmax==tier.xmax 时坍缩为零时长 |
| AM | `strip_edge_punctuation` ~983-985 | **Case 21**: 开头剥离增加 `not is_silence()` 检查，防 `<spN>` 被 `is_punct` 误判为标点 |
| AN | `process_one` ~4343,~4366 | **Case 22-A**: Phase 4 前后标点快照比对 → `_swallowed_puncts`（被吞标点追踪） |
| AO | `process_one` ~4665 | **Case 22-B**: 仅对 `_swallowed_puncts` 中确认被吞的标点, 若位置现为 `<spN>` → 替换恢复 |
| AP | `step_normalize_punct` ~680 | **Case 23**: Phase 2 增加 `is_hyphen_in_nvv` 检查，防 NVV 连字符被当作标点替换 |

---

## Case 1: MFA 尾静音被 snap 回词而非合并到标点 (jie2)

**日期**: 2026-07-17
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_snap_to_ctc`, `_inject_punctuation`
**触发样本**: 合成ria_15653, 词 `jie2` (8.03-8.51s)

### 现象

MFA 对齐后词尾的 `<eps>` 段在最终输出中重新被合并回前一词，而不是归入后续标点。

```
MFA 对齐:    jie2[8.03-8.27]  <eps>[8.27-8.52]  shi4[8.52-8.70]
修复前输出:  jie2[8.03-8.51]  dur=475ms  ，[8.51-8.52]
修复后输出:  jie2[8.03-8.27]  dur=240ms  ，[8.27-8.52]
```

### 根因链

1. **Rule 3 误判**: `ctc_dur(495ms) > mfa_dur(240ms) * 2 → use_mfa=False`。CTC 给 jie2 标了 495ms (包含尾静音)，MFA 正确切分 jie2=240ms + eps=250ms。但 2x 比例检查把"MFA 切掉了尾静音"误判为"词被压缩了"。
2. **`has_mfa_phone_evidence` 漏检**: 音素 `ie2` [8.10-8.51] 跨越了争议区 [8.27-8.505]，但检测未纳入。
3. **中间点未保护**: 即使 `use_mfa=True`，当 `end_diff > 0.15` 时取 CTC/MFA 中间点 (8.39)，没有检测尾静音+标点场景。
4. **微间隙合并 `kind` 不匹配**: `_snap_to_ctc` 插入的静音用 `kind="gap"`，但 `_inject_punctuation` 的合并规则仅匹配 `kind="word"`。

### 修改点

**A. Rule 3 绕过 (ratio_skip)**
**B. 中间点保护 (keep_mfa_end)**
**C. gap kind 扩展**
**D. Pattern (b) 缩进修复**
**G. has_trailing_sil xmax→xmin**

### 验证方法

```python
# 预期: jie2 dur ≈ 240ms (非 475ms), 逗号吸收了尾静音
words_tier["jie2"].duration < 0.30
words_tier after jie2: is_silence or is_punct
```

### 关联样本

- `合成ria_15653` jie2 → 逗号
- `合成ria_15653` le5 → 逗号

---

## Case 2: Phase 1 静音合并 vs Phase 3 Rule 3 顺位冲突 (er4)

**日期**: 2026-07-17
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `merge_short_silences`, `_snap_to_ctc`
**触发样本**: 合成ria_15653, 词 `er4` / `zhong3` (9.58s)

### 现象

MFA 正确切分 `zhong3 + <eps> + er4`，但最终输出 zhong3 350ms、er4 仅 50ms：

```
MFA aligned:  zhong3[9.23-9.39]  <eps>[9.39-9.58]  er4[9.58-9.64] dur=60ms
Phase 1:      zhong3[9.23-9.58]                    er4[9.58-9.64]
修复前:       zhong3[9.23-9.58]  dur=350ms         er4[9.58-9.63] dur=50ms
修复后:       zhong3[9.23-9.39]  dur=160ms         er4[9.39-9.63] dur=240ms
```

### 根因链

1. **Phase 1** `merge_short_silences`: 能量条件满足，`<eps>` 被合入 zhong3
2. **Rule 3**: er4 `ctc_dur(180) > mfa_dur(60) * 2` → snap 到 CTC
3. **重叠防护 (旧)** 将 er4 推后到 prev_end (9.58)，压成 50ms

### 修改点

**E. 重叠防护 prev_was_silence_extended** — 前词因静音延伸时缩短前词而非推后当前词
**F. 移除 gap_was_merged** — 能量分析否决

### 能量验证

```
9.23-9.38s: zhong3 RMS 0.016→0.001（音節结束）
9.38-9.46s: 静音 RMS 0.0002-0.001
9.47s:      er4 起振 RMS 0.031→0.18
```

CTC 锚点 er4 [9.45-9.63] 更接近真实。MFA `ong3` 音素 310ms 过度延伸。

### 关联样本

- `合成ria_15653` zhong3 → er4

---

## Case 3: 跨词界 eps 被 xmax 上限检查漏掉 (le5)

**日期**: 2026-07-17
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_snap_to_ctc`
**触发样本**: 合成ria_15653, 词 `le5` (6.27s)

### 现象

同 Case 1 pattern (a) — `le5 + <eps> + ，` — 但修复未生效：

```
MFA aligned:  le5[6.36-6.45]  <eps>[6.45-6.77]  kuai4[6.77-6.94]
修复前:       le5[6.27-6.70] dur=425ms  ，[6.70-6.77] dur=75ms
修复后:       le5[6.36-6.45] dur=90ms   ，[6.45-6.77] dur=320ms
```

### 根因

`has_trailing_sil` 的 `iv.xmax <= ctc_end + 0.05` 要求整个静音段在 CTC 范围内。le5 的 `<eps>` [6.45-6.77] 的 `xmax=6.77 > ctc_end+0.05=6.745`（eps 跨到了 kuai4 的区域），条件失败。

### 修改点

**G. `iv.xmax <=` → `iv.xmin <`** — 两处 has_trailing_sil 检查

### 关联样本

- `合成ria_15653` le5 → 逗号

---

## Case 4: 短 NVV 强制 CTC 导致前词被裁短 (ti2 / BREATHING)

**日期**: 2026-07-17
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_snap_to_ctc`
**触发样本**: 合成ria_13714, 词 `ti2` + `BREATHING` (2.16s)

### 现象

BREATHING (NVV) 强制用 CTC [2.31-2.37] (60ms)，其 start 与 ti2 MFA end (2.37) 重叠，NVV 重叠规则缩短前词：

```
MFA:    ti2[2.16-2.37] dur=210ms   BREATHING[2.37-2.45] dur=80ms
CTC:    ti2[2.13-2.31]             BREATHING[2.31-2.37] dur=60ms
修复前: ti2[2.16-2.31] dur=145ms   BREATHING[2.31-2.37] dur=60ms
修复后: ti2[2.16-2.37] dur=210ms   BREATHING[2.37-2.45] dur=80ms
```

### 根因

Rule 1 对所有 NVV 无条件设 `use_mfa=False`。但 NVASR 的短 NVV 检测 (< 100ms) 可能是噪声误检，CTC 锚点不可靠，反挤占相邻词边界。

### 修改点

**H. Rule 1 — NVV 短时长例外** (~line 2484)

```python
# 修改前: 所有 NVV 无条件 use_mfa=False
if is_nvv_token(mfa_iv.text) or is_english_token(mfa_iv.text):
    use_mfa = False

# 修改后: NVV CTC 时长 < 100ms 时保留 MFA 边界
if is_nvv_token(mfa_iv.text):
    use_mfa = (ctc_end - ctc_start) < 0.10
elif is_english_token(mfa_iv.text):
    use_mfa = False
```

### 修改点

**I. `_refine_boundaries_by_energy` — 词尾能量延伸** (~line 2320)

当词的元音衰减能量延续到紧邻的 NVV 区间（如 BREATHING），用能量分析将词尾延伸到真正的能量下跌点。仅限 NVV，不碰 silence/punct。保护 NVV 最小 40ms。

```
ti2:  MFA end=2.37 → 能量延伸 → 2.41
```

### 关联样本

- `合成ria_13714` ti2 → BREATHING (60ms NVV, ti2 从 2.37 延伸到 2.41)

---

## Case 5: MFA 词边界过晚——能量谷底在词首之前 (na2)

**日期**: 2026-07-17
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_refine_boundaries_by_energy`
**触发样本**: 合成ria_01251, 词 `na2` (1.51s)

### 现象

MFA 将 `shi4` 拆成 `<eps> + shi4`，导致 `na2` 的 start 被推到 1.51s。能量显示两个音节之间的谷底在 1.455s。

```
修复前: shi4[1.22-1.51]  na2[1.51-1.58] dur=70ms
修复后: shi4[1.22-1.455] na2[1.455-1.58] dur=125ms
```

### 根因

MFA 的 `<eps>` [1.22-1.45] 吞掉了 `shi4` 的声母 `sh`，Phase 1 把它合给了 `zhen1`。剩余 `shi4` 只有韵母 1.45-1.51，`na2` 被推到 1.51。CTC 锚点 (na2 start=1.41) 偏早。能量谷底在 1.455。

### 修改点

**J. `_refine_boundaries_by_energy` — 词首前拉** (~line 2320)

处理相邻两词时，在边界前 120ms 搜索能量谷底。约束：深谷（< 50% 峰值）、局部极小、后跟上升能量、前词 ≥ 80ms、拉动 25-80ms。

### 关联样本

- `合成ria_01251` na2 (start 1.51→1.455)

---

## Case 6: 静音段延伸被 end-trimming 回截 (ji2)

**日期**: 2026-07-17
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_refine_boundaries_by_energy`
**触发样本**: 合成ria_36502, 词 `ji2` (11.89s)

### 现象

ji2 后面有 250ms 死静音，应延伸到 12.44s。但 end-extension 延伸后被 end-trimming 截回。

```
修复前: ji2[11.89-12.19] dur=300ms
修复后: ji2[11.89-12.445] dur=555ms  he2[12.445-12.505]
```

### 根因链

1. **`is_punct("<eps>")` 返回 True**：`<eps>` 被误分类为标点，end-extension 跳过静音段
2. **onset 阈值 0.004 太高**：检测不到 he2 的轻辅音 /h/ 起振
3. **延伸后 intervals 重复**：`intervals[i+2]`（旧 he2）未删除，与新的 `intervals[i+1]`（移位 he2）重叠
4. **end-trimming 回截**：延伸后 ji2 尾部是静音，end-trimming 检测到尾部无能量 → 截回到 12.24

### 修改点

**K1.** `is_punct` 前先检查 `is_silence`：`if is_punct(next_iv.text) and not is_silence(next_iv.text): continue`

**K2.** onset threshold: `max(baseline * 3.0, 0.0015)` (原 `max(baseline * 4.0, 0.002)`)

**K3.** 延伸时吸收 silence 后标记 `intervals[i+2]` 为零时长占位，末尾过滤掉

**K4.** end-trimming 不变，改为 `_extended_indices` 集合保护被延伸过的词：`if i in _extended_indices: continue`

**K5.** punct 检查范围从 `gap_end+0.05` 扩大到全区间（含后续词）：避免标点落在 silent run 结束点之后被漏掉

**K6.** `_refine_boundaries_by_energy` 新增 `punct_entries` 参数，调用处传入 CTC 标点数据

### 关联样本

- `合成ria_36502` ji2 (end 12.19→12.445)

---

## Case 7: pinyin_phones 首音素与词界间隙 (ru2→r, NVV 邻接导致)

**日期**: 2026-07-23
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `build_pinyin_phones_tier`
**触发样本**: 合成ria_00001, 词 `ru2` (6.805s), pinyin_phones tier interval 54 "r"

### 现象

输出 TextGrid 中 pinyin_phones tier 在 NVV token `<UHM>` 与后邻词 `ru2` 的首音素 `r` 之间存在 5ms 间隙：

```
words tier:        <UHM>[6.250-6.805]  ru2[6.805-6.870]
pinyin_phones:     <UHM>[6.250-6.805]  (5ms gap)  r[6.810-6.840]  u2[6.840-6.870]
hanzi tier:        <UHM>[6.250-6.805]  如[6.805-6.870]
```

words/hanzi tier 连续无间隙，但 pinyin_phones tier 在 6.805-6.810 存在 5ms uncovered gap。
`r` 音素起点 (6.810) 比词界 (6.805) 晚 5ms。

**test_en_mfa 中也存在相同模式的间隙** (`合成ria_37435` 3.925s 处 5ms 间隙)。

### 根因链

1. **CTC 锚点**: `ru2` 原始 CTC 边界 [6.750-6.870]，`UHM` (NVV) 原始边界 [6.390-6.750]
2. **MFA fine_tune**: NVV token 无 MFA 声学模型，fine_tune 让 `UHM` 边界浮动到 [6.250-6.805]，将 `ru2` 词首从 6.750 推到 6.805
3. **MFA 音素对齐**: 在 `ru2` 词内，MFA 将首音素 [ɻ] (对应 pinyin `r`) 对齐到 6.810（比词首晚 5ms）。5ms 在 MFA 10ms 帧精度内，属 fine_tune 边界与音素对齐的残余偏差
4. **_snap_to_ctc** (Phase 3.A): `ru2` 非 NVV/English，MFA 边界被信任 (`use_mfa=True`)，音素保持 MFA 原始位置不变。词界 6.805，首音素仍 6.810
5. **`build_pinyin_phones_tier`** (Phase 1, line 533): 构建首音素时使用 `word_phones[0][0]`（MFA 音素的实际起点 6.810），而非词界 `w_iv.xmin` (6.805)。间隙被原样保留

### 修改点

**Q. `build_pinyin_phones_tier` — 逻辑修复：首/末音素无条件 snap 到词界** (~line 533, 539)

```python
# 修改前 — 音素边界来自 MFA phones tier（独立于 words tier）
new_intervals.append(Interval(word_phones[0][0], word_phones[0][1], dict_phones[0]))
final_end = word_phones[-1][1]

# 修改后 — 音素边界以 words tier 为权威
new_intervals.append(Interval(w_iv.xmin, word_phones[0][1], dict_phones[0]))
final_end = w_iv.xmax
```

**R. Phase 3.B 后同步 pinyin_phones** (~line 3708)

`_refine_boundaries_by_energy` 只接受/返回 `words_tier`，不碰 `pp_tier`。
能量调整后 words 边界变了但 pinyin_phones 没变——这是不同步的根源。
修改：Phase 3.B 更新 words 后，立即从 `phones_tier` + 新 `words_tier` 重建 `pinyin_phones`。

**S. Phase 3.5 后同步 pinyin_phones** (~line 3760)

`_apply_en_phones` 修改 `phones_tier`（注入英语 MFA 音素），但 `pinyin_phones` 未更新。
修改：英语音素注入后立即重建 `pinyin_phones`。

**T. words tier 连续性清理** (~line 3118, `_snap_to_ctc` 内)

MFA 对齐后词间可能有 ≤5ms 微小间隙（帧精度残余），`_snap_to_ctc` 的
`actual_gap > 0.005` 阈值漏掉了这些间隙。修改：构建完所有 word intervals
后统一扫描，将 ≤5ms 间隙吸收到前词尾部，确保 words tier 自身连续。
下游 tier（hanzi、pinyin_phones）依赖此不变式。

**架构原则：四个修改实现了同一目标——三个边界轨道 (words/hanzi/pinyin_phones)
始终以 words tier 为唯一权威，任何修改 words 或 phones 的操作必须立即同步
重建 pinyin_phones。**

### 验证方法

```python
# 首音素必须对齐词界
for w in words_tier:
    first_phone = first_non_sil_phone_in(w)
    assert abs(first_phone.xmin - w.xmin) < 0.002
    last_phone = last_non_sil_phone_in(w)
    assert abs(last_phone.xmax - w.xmax) < 0.002
```

### 关联样本

- `合成ria_00001` ru2 (6.805s → 6.726s after adjust, 词首间隙 5ms → 0ms)
- `合成ria_00001` ru2→guo3 (词间间隙 4ms → 0ms)
- `合成ria_37435` ian4→i4 (词间间隙 5ms, 修复 Q 後归零)

### 补充修改 (2026-07-17)

**L. start pull-back 搜索窗 120ms→80ms** (~line 2339)

120ms 窗口让 `max_rms` 被 50ms 外的 li4 元音峰值（RMS 0.059）污染，he2 元音衰减（RMS 0.004）被误判为"深谷"。缩短到 80ms 后 max_rms 仅覆盖局部邻域，消除了误判。

---

## 完整修改审查 (2026-07-17)

### `_snap_to_ctc` (Phase 3.A)

| 修改 | 行 | 状态 | 风险 |
|------|-----|------|------|
| A. ratio_skip (a) 尾静音+标点 | ~2520 | 稳定 | 低 |
| A. ratio_skip (b) 可见eps | ~2541 | 稳定 | 低 |
| B. keep_mfa_end 中间点保护 | ~2569 | 稳定 | 低 |
| D. Pattern (b) 缩进修复 | ~2541 | 稳定 | 已修复 |
| E. prev_was_silence_extended | ~2598 | 稳定 | 低 |
| G. has_trailing_sil xmax→xmin | ~2526,~2572 | 稳定 | 低 |
| H. NVV 短时长 <100ms 用MFA | ~2484 | 稳定 | 中：仅限短NVV |
| M. SILENCE_GAP_SNAP 有标点时跳过 | ~2914 | 稳定 | 低 |
| N. silence-adjacent 词首前拉 | ~2321 | 稳定 | 低：(silence→word only, onset_peak>0.002) |
| O. end-trimming 移除 English 豁免 | ~2623 | 稳定 | 低 |
| P. prev_was_silence_extended >100ms | ~2598 | 稳定 | 低 |

### `_inject_punctuation` (Phase 3.C)

| 修改 | 行 | 状态 | 风险 |
|------|-----|------|------|
| C. gap kind 扩展 "word"→("word","gap") | ~1877 | 稳定 | 低 |

### `_refine_boundaries_by_energy` (Phase 3.B)

| 修改 | 行 | 状态 | 风险 |
|------|-----|------|------|
| I. 词尾能量延伸 (NVV+静音) | ~2396 | 稳定 | 低 |
| I. NVV 前向延伸 | ~2518 | 稳定 | 低 |
| J. 词首前拉 (start pull-back) | ~2320 | 稳定 (L修后) | 低：80ms窗+0.003下限 |
| K1. is_punct 前先查 is_silence | ~2415 | 稳定 | 低 |
| K2. onset threshold 降低 | ~2483 | 稳定 | 低 |
| K3. 延伸后旧interval占位清除 | ~2550 | 稳定 | 低 |
| K4. _extended_indices 保护 | ~2396,~2623 | 稳定 | 低 |
| K5. punct 全范围检查 | ~2440 | 稳定 | 低 |
| K6. punct_entries 参数传递 | ~2233,~3624 | 稳定 | 低 |

### `build_pinyin_phones_tier` (Phase 1)

| 修改 | 行 | 状态 | 风险 |
|------|-----|------|------|
| Q. 首/末音素无条件snap到词界（逻辑修复） | ~533,~539 | 新增 | 低：以words边界为权威 |

### `_snap_to_ctc` (Phase 3.A)

| 修改 | 行 | 状态 | 风险 |
|------|-----|------|------|
| T. words tier 连续性清理（≤5ms间隙吸收） | ~3118 | 新增 | 低：MFA帧精度级别 |

### `process_one` Phase 3.B / 3.5

| 修改 | 行 | 状态 | 风险 |
|------|-----|------|------|
| R. Phase 3.B 后同步 pinyin_phones | ~3708 | 新增 | 低 |
| S. Phase 3.5 后同步 pinyin_phones | ~3760 | 新增 | 低 |

### `load_en_phones` / Phase 3.5

| 修改 | 行 | 状态 | 风险 |
|------|-----|------|------|
| 自动检测 en_phones_dir | ~3213 | 稳定 | 低 |
| 缺失告警 | ~3508 | 稳定 | 低 |

### 已知风险项

1. **NVV 短时长例外 (H)**：BREATHING<100ms 用 MFA 边界，可能不适于其他短 NVV
2. **静音延伸 (K) 与 end-trimming 的交互**：依赖 `_extended_indices` 精确保护
3. **intervals 列表顺序**：多次原位修改 intervals[i]/[i+1]/[i+2] 可能导致乱序，需增加排序保护

## Case 8: 标点间隙中的 <sp0> 因 has_punct 跳过合并 → mid_sp 误报

**日期**: 2026-07-27
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `handle_unexpected_silences`, `process_one` (mid_sp 检测)
**触发场景**: MFA 在标点与后邻词之间插入与后词同时开始的 `<sp0>` artifact

### 现象

words tier 中，标点后面出现 15ms 的 `<sp0>`，与下一个词同时开始：

```
words tier:
  [6.540-6.630] le5        (90ms)
  [6.630-6.660] ，          (30ms)
  [6.660-6.675] <sp0>       (15ms)  ← MFA artifact
  [6.660-6.820] wei4        (160ms) ← 与 <sp0> 同时开始
```

`<sp0>` 与 `wei4` 同时开始 (6.660)，说明这是 MFA 的对齐 artifact。15ms 的间隙没有任何语义意义。

### 根因链

1. MFA 在 `,` 和 `wei4` 之间插入 `<sp0>` [6.660-6.675]，`<sp0>.xmin == wei4.xmin`
2. `tg_word_idx` 过滤标点 → 内容词 `le5`, `wei4`
3. `gap_sil` 正确捕获 `<sp0>`, `gap_punct` 检测到逗号
4. **`has_punct` 短路跳过** (旧 L637): `if sil_label is None or has_punct: continue` → `<sp0>` 未被合并
5. `mid_sp` 检测命中 → 文件被误滤

### 修改点

**U. `handle_unexpected_silences` — `<sp0>` 无条件合并** (~line 636-641)

修改前:
```python
if sil_label is None or has_punct:
    continue
```

修改后:
```python
if sil_label is None:
    continue
if sil_label == "<sp0>":
    pass  # Always merge <sp0> regardless of punctuation
elif has_punct:
    continue  # <sp1-3> + punct: skip (handled by absorb pass)
elif sil_label in ("<sp1>", "<sp2>", "<sp3>"):
    ...
```

**V. `handle_unexpected_silences` — has_punct 时 `<sp0>` 合并到标点** (~line 668-693)

当 `has_punct=True` 时，`<sp0>` 不是合并到前词（会覆盖标点），而是查找邻近的标点间隔，扩展其 `xmax` 吸收 `<sp0>`，并从三个 tier 中删除 `<sp0>`。

```
修复前: le5[6.540-6.630] ，[6.630-6.660] <sp0>[6.660-6.675] wei4[6.660-6.820]
修复后: le5[6.540-6.630] ，[6.630-6.675]              wei4[6.660-6.820]
```

### 验证方法

```python
# words tier 中不应残留标点后的孤立 <sp0>
for i, iv in enumerate(words_tier.intervals):
    if i > 0 and iv.text.strip() == "<sp0>":
        prev = words_tier.intervals[i - 1]
        assert not is_punct(prev.text.strip()), \
            f"<sp0> after punct not merged: {prev.text} → <sp0>"
```

### 关联样本

- 外部项目: `le5 → ， → <sp0> → wei4`（6.540-6.820s 区间）

---

## Case 9: NVV 后的标点+静音链未被吸收 → mid_sp 误报

**日期**: 2026-07-27
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `absorb_nvv_trailing` (新增), `absorb_silence_into_punct` (新增)
**触发场景**: NVV 后跟标点+静音链，标点仅 5ms，`<sp2>` 695ms 成为孤立句中静音

### 现象

```
words tier:
  [9.270-9.745] zhi4
  [9.745-9.81]  <LAUGHTER>  (65ms, NVV)
  [9.81-9.815]  ！           (5ms)   ← 标点
  [9.815-10.51] <sp2>        (695ms) ← 孤立静音
  [10.51-10.65] bie2
```

实际音频中，9.81-10.51 是笑声尾音——它不是静音，是 NVV 的一部分。MFA 因无法对 NVV 做声学建模而将其标记为标点+`<sp2>`。

### 根本原因

MFA 无法对两类 token 做声学建模——**NVV** 和**标点**。它们在 MFA 的声学模型里没有对应 phone：

| Token | MFA 行为 | 产生的 artifact |
|------|---------|---------------|
| NVV (`<LAUGHTER>`, `<BREATHING>`, …) | 保留占位符，边界不精炼 | 后随的标点+静音残留 |
| 标点 (`，`, `！`, `…`, …) | 时长压缩到接近 0 | 后随的 `<sp>` 孤悬 |

CTC 预对齐给了初始边界但不准，MFA 无法精炼，postprocess 必须兜底清理。

### 根因链

1. **CTC prealign**: NVV token 无 phone 序列 → CTC 无法分配帧数 → NVV 只分到 65ms，剩余给了 `！` (5ms) 和 `<sp2>` (695ms)
2. **MFA align**: NVV 无 phone 模型 → 占位符保留，边界不精炼 → 相邻未建模段落变成 `<sp2>`
3. **handle_unexpected_silences**: `has_punct=True` → `<sp2>` 被跳过（合理——有标点的长静音不应标记为 unexpected）
4. **旧代码无补救**: 直到 `mid_sp` 检测前，无代码吸收此链
5. **mid_sp**: 孤立的 `<sp2>` → 文件误滤

### 修改点

**W1. 新增 `absorb_nvv_trailing` — Pass 1: NVV 吸收标点+静音链** (~line 758-851)

NVV 向右吞掉连续的标点+静音，直到下一个内容词。将 NVV 的 `xmax` 延伸到下一个实词的 `xmin`。

```
修复前: <LAUGHTER>[9.745-9.81] ！[9.81-9.815] <sp2>[9.815-10.51] bie2[10.51-10.65]
修复后: <LAUGHTER>[9.745-10.51]                                    bie2[10.51-10.65]
```

同步从 phones 和 pinyin_phones tier 删除被吸收的 `<spN>`（标点无 phone 条目无需处理）。

**W2. `absorb_silence_into_punct` — Pass 2 (兜底): 标点吸收残余 `<spN>`** (~line 854-920)

处理未被 NVV 吸收的残余场景（如无 NVV 时的 `标点 → <spN>`）。扩展标点的 `xmax` 吸收紧随其后的 `<spN>`。

**调用顺序** (~line 4031-4039): Phase 4 中 D → D2(`absorb_nvv_trailing`) → D3(`absorb_silence_into_punct`) → E:

```python
# D. handle_unexpected_silences (gap-level <sp0> merge)
# D2. absorb_nvv_trailing (NVV absorbs punct+silence chain)
# D3. absorb_silence_into_punct (fallback: punct absorbs residual <spN>)
```

### 与 Case 8 的关系

| | Case 8 | Case 9 |
|------|------|------|
| 静音类型 | `<sp0>` (< 0.2s) | `<sp1-3>` (≥ 0.2s) |
| artifact 来源 | 标点后的 MFA 帧精度残余 | NVV 无法声学建模 |
| 旧行为 | 残留 → mid_sp | 残留 → mid_sp |
| 修复位置 | `handle_unexpected_silences` (gap 级别) | `absorb_nvv_trailing` + `absorb_silence_into_punct` |
| 修复策略 | `<sp0>` 无条件合并 | Pass 1: NVV 吞链; Pass 2: 标点兜底 |

### 验证方法

```python
# NVV 后不应残留标点+<spN> 链
for i, iv in enumerate(words_tier.intervals):
    if is_nvv_token(iv.text) and i + 1 < len(words_tier.intervals):
        nxt = words_tier.intervals[i + 1].text.strip()
        assert not (is_punct(nxt) or (is_silence(nxt) and nxt)), \
            f"NVV trailed by punct/sil: {iv.text} → {nxt}"

# 标点后不应有孤立的 <spN>
for i in range(len(words_tier.intervals) - 1):
    cur = words_tier.intervals[i].text.strip()
    nxt = words_tier.intervals[i + 1].text.strip()
    assert not (is_punct(cur) and is_silence(nxt) and nxt), \
        f"<spN> after punct not absorbed: {cur} → {nxt}"
```

### 关联样本

- 外部项目: `zhi4 → <LAUGHTER> → ！ → <sp2> → bie2` (9.270-10.65s)

---

### 待处理

- **37443 yu2**: 目标 end=0.78s，当前 end=0.81s。yu2 尾部 (0.75-0.78) 有明显能量衰减，但 gang1 辅音起振 (0.795s, RMS 0.022) 落在 yu2 边界内，tail_rms gate 阻止了裁剪。end-trimming 无法区分"本词元音衰减"和"后词辅音起振"。需多词边界检测或能量谷底分割。

### 已解决

- **37435 jiu4**：修改 P 修复。jiu4 [10.39-10.49]，目标 [10.38-10.50]。
- **37434 ru2**：silence-adjacent 词首前拉 (N)。冒号 [6.80-7.355]，ru2 [7.355-7.50]。
- **Case 8 (标点间隙<sp0>残留)**：修改 U+V。`handle_unexpected_silences` 中 `<sp0>` 无条件合并。
- **Case 9 (NVV后标点+静音链)**：修改 W1+W2。新增 `absorb_nvv_trailing` (D2) + `absorb_silence_into_punct` (D3 兜底)。
- **Case 10 (Hanzi tier拼音残留)**：修改 X+Y+Z。`_word_matches` 元音约束 + `_build_hanzi_tier` 重写为顺序映射。
- **Case 11 (cjk_mismatch时序bug + 缺直接拼音扫描)**：修改 AA1+AA2。检查移至路径决策前 + 新增 `hanzi_pinyin` 直接扫描。
- **Case 15 (_extract_word_chars NVV拆分)**：修改 AJ。`<` `>` flush 处理，防 NVV token 被拆分为独立单元。

---

## Case 11: cjk_mismatch / hanzi_pinyin 检查在路径决策之后执行 → 文件未被重定向

**日期**: 2026-07-27
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (filtering section)
**触发场景**: hanzi tier 中存在拼音残留或 CJK 字符错位，但文件仍写入 `output_dir` 而非 `filtered_dir`

### 现象

当 hanzi tier 构建出错（如 Case 10 的 qie4 残留、CJK 字符丢失），现有的 `cjk_mismatch` 检查能检测到问题，但文件仍被写入 `output/` 目录而不是 `filtered/` 目录：

```
report: { "status": "ok", "cjk_details": { "raw_count": 2, "hanzi_count": 1 } }
文件路径: output/xxx.TextGrid     ← 应该在 filtered/ 下！
```

### 根因链

1. **路径决策在检查之前**：`filter_reasons` → `out_path` 的判断在 `cjk_mismatch` 检查之前执行
2. **`if not filter_reasons` 守卫过严**：`cjk_mismatch` 检查只在无其他过滤原因时执行，有其他原因时跳过
3. **缺少直接拼音残留扫描**：无 `is_pinyin_syllable()` 对 hanzi tier labels 的直接检查，仅依赖间接的 CJK 字符计数对比

### 时序对比

```
修复前:
  4717: if filter_reasons: out_path = filtered_dir     ← 此时 filter_reasons 为空
  4726: else: out_path = output_dir                     ← 文件去 output/
  ...
  4738: if not filter_reasons:                          ← 进入检查
  4752:     filter_reasons.append("cjk_mismatch")       ← 检测到了，但太晚！
  4760: write_textgrid(new_tg, out_path)               ← 写入 output/，不是 filtered/

修复后:
  4717: hanzi tier integrity checks                     ← 先检查
  4731:     filter_reasons.append("hanzi_pinyin")        ← 检测到拼音残留
  4746:     filter_reasons.append("cjk_mismatch")         ← 检测到 CJK 不匹配
  4751: if filter_reasons: out_path = filtered_dir       ← 正确重定向！
```

### 修改点

**AA1. `process_one` — Hanzi tier 完整性检查移至路径决策之前** (~line 4717-4749)

两处检查（拼音残留 + CJK 覆盖）从路径决策之后（原 `if not filter_reasons:` 守卫内）移至路径决策之前，移除 `if not filter_reasons` 守卫，确保所有文件都经过检查。

**AA2. `process_one` — 新增直接拼音残留扫描 `hanzi_pinyin`** (~line 4724-4734)

对 hanzi tier 每个 interval 做 `is_pinyin_syllable()` 扫描，检测到拼音残留时添加 `hanzi_pinyin` 过滤原因：

```python
pinyin_labels: list[str] = []
for iv in hanzi_tier_final.intervals:
    label = iv.text.strip()
    if label and is_pinyin_syllable(label):
        pinyin_labels.append(label)
if pinyin_labels:
    filter_reasons.append("hanzi_pinyin")
    report.setdefault("hanzi_pinyin", {})["count"] = len(pinyin_labels)
    report["hanzi_pinyin"]["labels"] = pinyin_labels[:20]
```

### 两个检查的关系

| 检查 | 检测方式 | 覆盖场景 |
|------|---------|---------|
| `hanzi_pinyin` (新增) | 直接 `is_pinyin_syllable()` on hanzi labels | qie4 残留、任何拼音 token 未转换 |
| `cjk_mismatch` (修复) | raw_text CJK 序列 vs hanzi CJK 序列字符串比对 | CJK 丢失、错位、多余 CJK |

两者互补：
- `hanzi_pinyin` 能直接检测到拼音残留这一**根因**
- `cjk_mismatch` 能检测到 CJK 字符序列的任何不一致（包括但不限于拼音残留）

### 验证方法

```python
# 拼音残留检测
for iv in hanzi_tier.intervals:
    assert not is_pinyin_syllable(iv.text.strip()), \
        f"hanzi_pinyin: {iv.text} at [{iv.xmin}-{iv.xmax}]"

# CJK 覆盖检测
assert raw_cjk == hanzi_cjk, \
    f"cjk_mismatch: raw={len(raw_cjk)} hanzi={len(hanzi_cjk)}"
```

### 关联样本

- 外部项目: `SURPRISE，OH 切片OP` — qie4 → 切 (Case 10 修复预防，Case 11 作为防御)

---

## Case 10: Hanzi tier 拼音残留 — NW 对齐 + 过度宽松匹配 → CJK 被跳过 (qie4/切)

**日期**: 2026-07-27
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_build_hanzi_tier`, `_word_matches`, `_finalise_textgrid`
**触发场景**: 参考文本包含多音字 + 短英文词相邻（如"切片OP"中 qie4 与 OH/OP 竞争）

### 现象

参考文本是混合中英文：

```
SURPRISE，OH 切片OP
```

最终 TextGrid 的 `hanzi` tier 里，唯独"切"变成了拼音 `qie4`，其他 token 都是正确的汉字/英文：

| words tier | hanzi tier (修复前) | hanzi tier (修复后) |
|-----------|-------------------|-------------------|
| SURPRISE-OH | SURPRISE | SURPRISE |
| qie4 | **qie4** ✗ | **切** ✓ |
| pian4 | 片 | 片 |
| OP | OP | OP |

### 完整链路

```
参考文本: … SURPRISE，OH 切片OP …
              │
              ▼
  ① chars_and_pinyin() — 逐字转拼音生成 .lab
     pypinyin.lazy_pinyin("切") → "qie4"
     （pypinyin 默认词典，"切"独立出现时选第四声 qie4）
     .lab 文件: SURPRISE-OH qie4 pian4 OP
              │
              ▼
  ② MFA 对齐 — 根据 .lab 拼音做强制对齐
     words tier: SURPRISE-OH | qie4 | pian4 | OP
              │
              ▼
  ③ _build_hanzi_tier() — 拼音词 → 汉字反向映射
     【旧代码】用 Needleman-Wunsch 全局对齐 + _word_matches()
              │
              ▼
  ④ _word_matches() 第 1074 行过度宽松匹配
     qie4 是拼音音节格式 → 匹配任意英文单词（代价 0）
     qie4 ↔ "OH" 代价 0；qie4 ↔ "切" 代价 0
     NW gap-first 回溯：优先消费 "OH"，把 qie4 配对给 OH
     "切" 变 reference-only gap → 被跳过
              │
              ▼
  ⑤ hanzi tier: SURPRISE | qie4 | 片 | OP
     "切" 丢失，"qie4" 作为 CTC-only 标签残留
```

### 根因链

1. **pypinyin 多音字默认声调错误**：`lazy_pinyin("切")` 在逐字模式下返回 `qie4`（第四声），而 "切片" 中应读 `qie1`（第一声）。声调错误意味着 pinyin "qie4" 与 CJK "切" 的精确匹配失败（`py[0] == c` 检查 `py == "qie4"` 而 `lazy_pinyin("切")` 也返回 `"qie4"` — 但这是关键：两边都是 qie4，所以精确匹配实际上通过了）。

    **更正**：实际根因不是声调不匹配，而是 `_word_matches()` 第 1074 行的无差别匹配：`return True` 让任何拼音音节匹配任何英文单词代价为 0。NW gap-first 回溯在代价相同时优先 gap（跳过 ref），选择了代价相同的 `qie4↔OH` 而非 `qie4↔切`。

2. **`_word_matches()` 过度宽松**：拼音音节格式 `[a-z]+[1-5]` 无条件匹配任何英文单词（代价 0），不检查参考词长度或元音数。

3. **NW gap-first 回溯**：在 `dp[i][j] == dp[i-1][j] + 1`（CTC gap）和 `dp[i][j] == dp[i][j-1] + 1`（reference gap）代价相同时，优先选 CTC gap。当 `qie4↔OH` 和 `qie4↔切` 代价都为 0 时，gap-first 回溯选择消费 OH 而非 切。

### 触发场景分类

**场景 A — 多音字默认声调**：pypinyin 对"切"、"的"、"了"、"为"等字选错默认声调。声调错误本身不直接触发 bug（因为 _word_matches 用同样的 lazy_pinyin 做反向匹配），但声调不同让精确匹配代价变 1，不再严格优于宽松的拼音→英文匹配。

**场景 B — 拼音音节 + 短英文词相邻（直接触发）**：words tier 中拼音音节 token 与短英文 token（OH, OP, in, up）相邻，NW 的 gap-first 回溯在代价相同时优先消费英文词。

**场景 C — MFA 合并英文相邻词**：MFA 将相邻英文词合并（SURPRISE + OH → SURPRISE-OH），增加了 token 序列的不确定性。

**场景 D — MFA 丢弃 token（极端）**：上游丢弃 CJK 拼音 token，导致下游错位。

### 修改点

**X. `_word_matches` — 拼音→英文匹配增加元音约束** (~line 1069-1076)

修改前（旧）：
```python
# 旧：return True（任意拼音音节匹配任意英文词）
if len(c) >= 2 and c[-1].isdigit() and c[:-1].isalpha():
    return True
```

修改后（新）：
```python
# 新：仅当英文参考词 ≥2 个元音字母时才匹配
if len(c) >= 2 and c[-1].isdigit() and c[:-1].isalpha():
    vowel_count = sum(1 for ch in r if ch in 'aeiou')
    return vowel_count >= 2
```

效果：
- `qie4` 不再匹配 `OH`（1 个元音）✓
- `qie4` 不再匹配 `OP`（1 个元音）✓
- `ai4` 仍然匹配 `idol`（2 个元音）✓
- `rui4` 仍然匹配 `ria`（2 个元音）✓

**Y. `_build_hanzi_tier` — 重写：顺序 CJK 映射替代 NW 对齐** (~line 1173-1298)

核心思想：CJK 字符按顺序一一对应 MFA 拼音词位置，不依赖 pypinyin 声调准确性。

```
CJK 字符：拼音音节按顺序消费参考文本中的下一个 CJK 字
  不依赖 pypinyin 声调准确性
  不依赖任何字典
  qie4 → 下一个未使用的 CJK 字 → "切" ✓

英文/NVV：贪心子串匹配
  SURPRISE-OH → 消费 "SURPRISE" + "OH" 两个参考单元
  "li" → 匹配 "live"（子串）
```

关键改动：
- CJK 字符和英文词**分开处理**，使用独立的消费游标
- CJK：`is_pinyin_syllable(token)` → 从 `ref_cjk` 队列消费下一个字符
- 英文/NVV：`_alpha_text_matches(token, ref)` → 贪心消费匹配的 alpha 参考单元
- 标点：`is_punct()` → 原样透传，不消耗任何游标
- silence：保持 silence label 透传

**Z. `_alpha_text_matches` — Alpha token ↔ ref unit 匹配** (~line 1136-1170, 新增)

从旧 `_word_matches` 中抽取 alpha 匹配逻辑（去掉 CJK pinyin 精确匹配部分），用于 `_build_hanzi_tier` 的贪心消费。

### 防御性检测（新增）

当 words tier 中的拼音音节数量 ≠ 参考文本中的 CJK 字符数量时，向 `report["warnings"]` 输出 warning：

```
# 拼音 token 多于 CJK 字符（token 冗余）
"hanzi tier mismatch: 4 pinyin tokens vs 2 reference CJK chars
 — 2 pinyin token(s) fell back (no more CJK chars to consume)"

# CJK 字符多于拼音 token（字符丢失）
"hanzi tier mismatch: 2 pinyin tokens vs 3 reference CJK chars
 — 1 reference CJK char(s) were not assigned to any pinyin token"
```

### warnings 参数 threading

`_finalise_textgrid` 新增 `warnings` 参数 → 透传至 `_build_hanzi_tier`。两处调用点（`process_one` Phase 2 的 `_finalise_textgrid` 和 Phase 5 的直接 `_build_hanzi_tier` 调用）均传入 `report.get("warnings", [])`。

### 核心注意点

1. **`lazy_pinyin` 逐字模式的固有限制**：pypinyin 对每个字独立判断，无法利用上下文消歧。解决方案是让 hanzi tier 构建不依赖 pypinyin 的返回值。

2. **顺序映射的前提假设**：MFA words tier 中拼音音节的顺序和数量 = 参考文本中 CJK 字符的顺序和数量。正常 pipeline 中此前提始终成立。

3. **NW 对齐仍然保留**：`_align_word_sequences()` + `_word_matches()` 仍被 `_normalize_word_spellings()` 使用。修复只改了 `_word_matches` 中的一条规则（元音约束），保留了其他所有匹配逻辑。

4. **标点处理**：标点在 words tier 中作为独立 interval 存在时，原样透传不消耗游标。被 MFA 挤掉时对 CJK 顺序映射无影响。

5. **英文词合并与拆分**：合并（SURPRISE-OH → 两个参考词）通过贪心匹配依次消费。拆分（li ve → live）由第一个 token 消耗参考词，第二个变 CTC-only gap。

### 修改文件清单

| 文件 | 行号 | 改动 |
|------|------|------|
| postprocess_textgrids.py | ~924 | `_finalise_textgrid` 新增 `warnings` 参数 |
| postprocess_textgrids.py | ~961 | 向 `_build_hanzi_tier` 传递 warnings |
| postprocess_textgrids.py | ~1069-1076 | `_word_matches` 拼音→英文匹配增加元音约束 |
| postprocess_textgrids.py | ~1136-1170 | `_alpha_text_matches` 新增 |
| postprocess_textgrids.py | ~1173-1298 | `_build_hanzi_tier` 重写：顺序 CJK 映射 + 贪心 alpha 匹配 + 防御检查 |
| postprocess_textgrids.py | ~3967 | `process_one` 向 `_finalise_textgrid` 传递 `report["warnings"]` |
| postprocess_textgrids.py | ~4300 | `process_one` 向第二次 `_build_hanzi_tier` 传递 `report["warnings"]` |

### 验证方法

```python
# 拼音音节 token 必须映射到 CJK 字符，不能残留为拼音
for iv in hanzi_tier.intervals:
    assert not is_pinyin_syllable(iv.text.strip()), \
        f"Pinyin residue in hanzi tier: {iv.text}"

# CJK 字符计数应与拼音 token 计数一致
# (不一致时只在 warnings 中报告，不阻断)
```

### 关联样本

- 外部项目: `SURPRISE，OH 切片OP` (qie4 → 切 修复)

### 补充修改 (2026-07-28)

**warnings 重复修复** — `_finalise_textgrid` 调用不再传递 warnings。

根因：`process_one` 中两次调用 `_build_hanzi_tier`：
1. Phase 2（line ~3976，通过 `_finalise_textgrid`）— 构建的 hanzi tier 在 Phase 5 被替换
2. Phase 5（line ~4311）— 构建最终 hanzi tier

两处都传入 `report["warnings"]` 导致每条 mismatch 警告出现两次。
修复：Phase 2 的 `_finalise_textgrid` 不再传递 warnings（默认 `None`，跳过防御检查），
仅 Phase 5 的调用负责输出 warnings。

---

## Case 12: CTC snap 重叠修复产生倒置 interval + 尾静音未被最后标点吸收 (xia4/ta1)

**日期**: 2026-07-27
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_snap_to_ctc`, `_inject_punctuation`
**触发样本**: 花礼_不可言说之小礼猫跳船日_20250331_2331, segment 00060000 (clip0006_clip0000)

### 现象

参考文本末尾"你看一下他。"，实际音频中"他"字发音极短 (MFA 检测仅 0.06s)。Pipeline 输出中 `ta1`（他）从 words/hanzi tier 消失，尾静音 `<sp2>` 与前后 interval 重叠：

```
MFA 对齐:
  xia4[10.79-10.87] dur=0.08s  ta1[10.87-10.93] dur=0.06s  <sp2>[10.93-11.49]  。[10.95-11.565875]

CTC 预对齐:
  xia4[10.79-10.95] dur=0.16s   ta1[10.87-10.93] dur=0.06s

修复前 (有 bug):
  words:  xia4[10.79-10.95]  <sp2>[10.93-11.49]  。[10.95-11.565875]
  hanzi:  下                                    。     ← "他" 丢失!
  ↑ <sp2>.xmin(10.93) < xia4.xmax(10.95) 重叠!
  ↑ 。.xmin(10.95) < <sp2>.xmax(11.49) 重叠!

修复后 (预期):
  words:  xia4[10.79-10.95]  ta1[10.95-11.01]  。[11.01-11.565875]
  hanzi:  下                他                。
  ↑ 边界连续无重叠
```

### 根因链

**Bug 1 — 倒置 interval**:

1. **xia4（下）**：MFA 时长 0.08s，CTC 时长 0.18s → `ctc_dur > mfa_dur * 2`，触发 Rule 3 (`use_mfa=False`)，xmax 从 MFA 的 10.87 snap 到 CTC 的 10.95
2. **ta1（他）**：MFA 时长 0.06s，MFA 和 CTC 偏差均在 0.3s 阈值内 → `use_mfa=True`，保持 MFA 边界 10.87–10.93
3. **重叠检测**：`ta1.xmin(10.87) < xia4.xmax(10.95) - 0.002` → 触发重叠修复
4. **prev_was_silence_extended 分支不匹配**：xia4 不是 NVV，`prev_end(10.95) > prev_ctc_end(10.95) + 0.10` 为 False → 进入 else 分支：`word_start = prev_end = 10.95`
5. **word_end 未更新**：ta1 的 word_end 仍为 MFA 原始值 10.93
6. **倒置 interval**：ta1 = (10.95, 10.93) → `xmin > xmax`！
7. **静默丢弃**：`_inject_punctuation` 中 `if e > s` 过滤（~line 2098）将倒置的 ta1 丢弃，他 从 hanzi tier 消失

**Bug 2 — 尾静音未被吸收**:

1. ta1 被丢弃后，`_snap_to_ctc` 在 `prev_end(10.93)` 之后插入尾静音 `<sp2>(10.93-11.49)`
2. `_inject_punctuation` 最后标点逻辑：查找最后非静音词 → xia4，`punct_start = xia4.xmax = 10.95`
3. 条件 `m[0] < punct_start`：`<sp2>.xmin(10.93) < 10.95` → 被保留！
4. `<sp2>` 与 xia4 重叠 (10.93 < 10.95)，又与扩展后的 。重叠 (10.95 < 11.49)

### 修改点

**AC. `_snap_to_ctc` — 重叠修复后防倒置 interval 保护** (~line 3317)

修改前：
```python
            else:
                word_start = prev_end

        # ── Gap absorption (ORDER CRITICAL — do not reorder) ──
```

修改后：
```python
            else:
                word_start = prev_end

        # Guard against inverted intervals: when overlap fix pushes word_start
        # past word_end (prev word CTC-snapped longer than current word's MFA
        # end), extend word_end to preserve the word with at least its MFA
        # duration or a 30 ms floor.
        if word_end < word_start:
            word_end = word_start + max(mfa_dur, 0.030)

        # ── Gap absorption (ORDER CRITICAL — do not reorder) ──
```

效果：ta1 不再倒置，`word_end = 10.95 + max(0.06, 0.030) = 11.01`，保留原始时长。

**AD. `_inject_punctuation` — 最后标点保留条件从 xmin 改为 xmax** (~line 2277)

修改前：
```python
            elif m[0] < punct_start:
                new_merged.append(m)
```

修改后：
```python
            elif m[1] <= punct_start + 0.001:
                new_merged.append(m)
```

效果：`<sp2>.xmax(11.49) <= 10.95 + 0.001` → `11.49 <= 10.951` → False → 被最后标点吸收。`xia4.xmax(10.95) <= 10.951` → True → 保留。

### 触发条件（三个条件同时满足）

1. **前词被 CTC 快照拉长**：MFA 时长 << CTC 时长（超过 2× 比例阈值，Rule 3），`use_mfa=False`，xmax 延伸到 CTC 边界
2. **当前词保持 MFA 边界**：MFA 和 CTC 偏差均在阈值内，`use_mfa=True`
3. **当前词完全被前词覆盖**：`cur.xmin < prev.xmax`

### 关联样本

- `花礼_不可言说之小礼猫跳船日_20250331_2331` segment `00060000`：xia4(下) → ta1(他) → 。

---

## Case 13: NVV 大小写不一致 — MFA 小写 NVV 不被识别/包裹 → pinyin_phones 残留小写 (laughter)

**日期**: 2026-07-27
**涉及文件**: `scripts/postprocess_textgrids.py`, `scripts/finalize_textgrids.py`
**涉及函数**: `_finalize_textgrid`, `build_pinyin_phones_tier`
**触发样本**: segment 00150009 (花礼 pipeline), `<LAUGHTER>` 出现在 words/hanzi 但 pinyin_phones 残留小写 `laughter`

### 现象

`<LAUGHTER>` 在 words/hanzi/pinyin tier 中是大写并包裹 `<>`，但 pinyin_phones tier 中却是小写无括号：

```
raw_text (T1):   ...<LAUGHTER>！
pinyin (T2):     ... <LAUGHTER> ！
hanzi (T3):      [54] <LAUGHTER>
words (T4):      [54] <LAUGHTER>
pinyin_phones (T5): [95] laughter    ← 小写，无 <>，与其他 tier 不一致！
```

### 根因链

1. **MFA 输出小写**：MFA 将 OOV（含 NVV）统一输出为小写 `laughter`（非 `LAUGHTER`）
2. **`_NVV_PATTERN` 大小写敏感**：regex 仅匹配 `(?<![A-Z-])` + 大写 `NVV_NAMES`，小写 `laughter` 不匹配 → `_finalize_textgrid` 中 `_NVV_PATTERN.sub` 不生效
3. **CTC token 提供大写**：CTC 预对齐输出 `LAUGHTER`（大写），words tier 在某步被更新为大写，后续 `_finalize_textgrid` 能匹配并包裹
4. **`build_pinyin_phones_tier` 时机早**：在 words tier 被 CTC 修正前就已执行，复制了小写 `laughter`，后续不再重建
5. **`finalize_textgrids.py` 无防护**：直接用 `f"<{iv.text}>"` 包裹，保留原始大小写，且无双重包裹防护（若已包裹 `<>` 会变成 `<<LAUGHTER>>`）

### 修改点

**AE. `_NVV_PATTERN` — 添加 `re.IGNORECASE` + 扩展字符类** (~line 87-91)

修改前：
```python
_NVV_PATTERN = re.compile(
    r"(?<![A-Z-])("
    + "|".join(re.escape(name) for name in sorted(NVV_NAMES, key=len, reverse=True))
    + r")(?![A-Z-])"
)
```

修改后：
```python
_NVV_PATTERN = re.compile(
    r"(?<![A-Za-z-])("
    + "|".join(re.escape(name) for name in sorted(NVV_NAMES, key=len, reverse=True))
    + r")(?![A-Za-z-])",
    re.IGNORECASE
)
```

效果：`laughter`, `Laughter`, `LAUGHTER` 均被匹配，同时确保不被部分匹配（如 `slaughter`）。

**AF. `_finalize_textgrid` — 替换文本统一大写** (~line 141)

修改前：
```python
iv.text = _NVV_PATTERN.sub(r"<\1>", iv.text)
```

修改后：
```python
iv.text = _NVV_PATTERN.sub(lambda m: f"<{m.group(1).upper()}>", iv.text)
```

效果：无论 MFA 输出什么大小写，最终统一为 `<LAUGHTER>`。

**AG. `build_pinyin_phones_tier` — NVV 文本规范化** (~line 485-488)

修改前：
```python
# NVV token: one self-referential phone
if is_nvv_token(w_iv.text):
    new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, w_iv.text))
    continue
```

修改后：
```python
# NVV token: one self-referential phone — normalize to <UPPERCASE>
if is_nvv_token(w_iv.text):
    nvv_text = f"<{w_iv.text.strip().strip('<>').upper()}>"
    new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, nvv_text))
    continue
```

效果：pinyin_phones tier 中 NVV 始终以规范格式 `<UPPERCASE>` 写入，与 words tier 一致。`strip('<>')` 防双重包裹。

**AH. `finalize_textgrids.py` — NVV 规范化** (~line 50-54)

修改前：
```python
for tier in tg.tiers:
    for iv in tier.intervals:
        if iv.text and is_nvv(iv.text):
            iv.text = f"<{iv.text}>"
```

修改后：
```python
for tier in tg.tiers:
    for iv in tier.intervals:
        if iv.text and is_nvv(iv.text):
            # Normalize to <UPPERCASE> — strip existing brackets/case first
            # to avoid double-wrapping and mixed-case inconsistencies.
            cleaned = iv.text.strip().strip('<>').upper()
            iv.text = f"<{cleaned}>"
```

效果：独立脚本也输出一致的 `<UPPERCASE>` 格式，且不会 `<<LAUGHTER>>`。

### 验证方法

```python
# _NVV_PATTERN 大小写不敏感
assert _NVV_PATTERN.search("laughter")
assert _NVV_PATTERN.search("Laughter")
assert _NVV_PATTERN.search("LAUGHTER")
assert not _NVV_PATTERN.search("slaughter")  # 不部分匹配

# 替换统一大写
assert _NVV_PATTERN.sub(lambda m: f"<{m.group(1).upper()}>", "laughter") == "<LAUGHTER>"

# pinyin_phones NVV 规范化
assert normalize_nvv("laughter") == "<LAUGHTER>"
assert normalize_nvv("<LAUGHTER>") == "<LAUGHTER>"  # 无双重包裹

# finalize_textgrids 规范化
assert finalize_nvv("laughter") == "<LAUGHTER>"
assert finalize_nvv("Laughter") == "<LAUGHTER>"
```

### 关联样本

- `00150009` (花礼 pipeline): `<LAUGHTER>` — words/hanzi tier 大写包裹，pinyin_phones 小写无包裹

---

## Case 14: Hanzi tier NVV 括号污染 — `_build_hanzi_tier` alpha 分支未 strip `<>` → label 残留 `<SURPRISE-WA>`

**日期**: 2026-07-28
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_build_hanzi_tier`, `_alpha_text_matches`
**触发样本**: segment 00150015 (20241117 花礼 pipeline)

### 现象

Hanzi tier 中 `<SURPRISE-WA>` 残留了 NVV 括号包裹，且 `ai1` 未映射到汉字 `哎`：

```
words tier:     [2] <SURPRISE-WA>    [3] ai1
hanzi (修复前): [2] <SURPRISE-WA>    [3] ai1      ← 括号污染 + 拼音残留
hanzi (修复后): [2] SURPRISE-WA      [3] 哎        ← 正确
```

逗号时间在三个轨道一致（1.71–2.43），问题仅限于 hanzi tier 逗号之前的映射。

### 根因链

1. **`SURPRISE-WA` 是标准 NVV 名称**：MFA OOV 后 `_finalize_textgrid` 将其包裹为 `<SURPRISE-WA>`
2. **`_alpha_text_matches` 内部能 strip `<>`**：子串匹配 `"surprise-wa" in "<surprise-wa>"` 正常返回 True
3. **但 fallback 路径使用原始 `token`**（含 `<>`）：当 `ref_alpha` 为空或匹配失败时，`matched_refs = []`，label 回退为原始 token 文本 `<SURPRISE-WA>`（`<` 不是 alpha/hyphen，`all(c.isalpha() or c == '-' for c in token)` 失败 → 最终 `label = token`）
4. **`continue_match` 检查也使用原始 `token`**：`token.lower()` 为 `"<surprise-wa>"`，多词消费判断受 `<>` 干扰
5. **CJK 游标受影响**：alpha 未匹配成功 → `alpha_idx` 不推进 → `ai1` 可能错误匹配到 alpha ref 而非 CJK 字符

### 修改点

**AI. `_build_hanzi_tier` — Alpha 分支 strip `<>`** (~line 1246-1278)

修改前：
```python
        # ── English / NVV token → greedy match against alpha refs ──
        matched_refs: list[str] = []
        while alpha_idx < len(ref_alpha):
            ref_unit = ref_alpha[alpha_idx]
            if _alpha_text_matches(token, ref_unit):
                ...
                if next_ref in token.lower() or token.lower() in next_ref:
                    ...
        ...
        elif token.isascii() and all(c.isalpha() or c == '-' for c in token):
            label = token
        else:
            label = token
```

修改后：
```python
        # ── English / NVV token → greedy match against alpha refs ──
        # Strip angle brackets: _finalize_textgrid may have already
        # wrapped NVV tokens with < > before we run.  Matching and
        # fallback labels must use the clean form to avoid bracket
        # pollution in the hanzi tier and misaligned cursors.
        clean_token = token.strip('<>')
        matched_refs: list[str] = []
        while alpha_idx < len(ref_alpha):
            ref_unit = ref_alpha[alpha_idx]
            if _alpha_text_matches(clean_token, ref_unit):
                ...
                if next_ref in clean_token.lower() or clean_token.lower() in next_ref:
                    ...
        ...
        elif clean_token.isascii() and all(c.isalpha() or c == '-' for c in clean_token):
            label = clean_token
        else:
            label = clean_token
```

效果：
- `<SURPRISE-WA>` → `clean_token = "SURPRISE-WA"` → 匹配 + fallback 均使用无括号形式
- `all(c.isalpha() or c == '-' for c in "SURPRISE-WA")` → True → fallback label 为 `SURPRISE-WA`（而非 `<SURPRISE-WA>`）
- `continue_match` 检查使用 `clean_token` → 多词消费不受 `<>` 干扰

### 验证方法

```python
clean = token.strip('<>')
# 匹配
label = _alpha_match_and_fallback(clean, ref_alpha)
assert '<' not in label and '>' not in label  # 无括号污染
# CJK 游标正确
assert cjk_idx == expected  # ai1 → 哎 (CJK 正确消费)
```

### 关联样本

- `00150015` (20241117 花礼 pipeline): `<SURPRISE-WA>` + `ai1` → 逗号之前 hanzi 映射错误

### 本段全部修改汇总

| 段 | 问题 | Case | 修复 |
|---|------|------|------|
| 20250331/00060000 | `_snap_to_ctc` 倒置 interval → 他 丢失 | 12 | overlap fix: push word_end |
| 20250331/00060000 | `<sp2>` 未被末尾标点吸收 → 边界重叠 | 12 | `m[1] <= punct_start` |
| 20250202/00150009 | pinyin_phones `laughter` ≠ words `<LAUGHTER>` | 13 | case-insensitive NVV 包裹 + 统一大写 |
| 20241117/00150015 | hanzi 残留 `<SURPRISE-WA>` 和 `ai1` | 14 | strip `<>` + alpha 匹配用 `clean_token` |

---

## Case 15: `_extract_word_chars` 拆分 NVV 尖括号 — `<LAUGHTER>` → `[<, LAUGHTER, >]`

**日期**: 2026-07-28
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_extract_word_chars`, `_build_hanzi_tier`
**发现方式**: CROSS_CASE_ANALYSIS.md Risk 4.5 验证过程中发现

### 现象

`_extract_word_chars` 将 NVV token 的尖括号当作标点拆分：

```
_extract_word_chars("<LAUGHTER>你好")
修复前: ['<', 'LAUGHTER', '>', '你', '好']   ← NVV 被拆成 3 个单元
修复后: ['<LAUGHTER>', '你', '好']            ← 正确
```

### 根因链

1. **`<` 和 `>` 不在 alpha 字符集中**：`_extract_word_chars` 的 alpha buffer 条件为 `c.isalpha() or c == '-'`，未包含 `<>`
2. **尖括号被当作标点**：落到 `else` 分支 → flush buffer → 每个尖括号作为独立标点 entry
3. **`is_word_like` 过滤差异**：`"<"` 不是 word-like（无 alpha/CJK/digit）→ 被过滤；`"LAUGHTER"` 是 word-like（以 alpha 开头）→ 进入 `ref_alpha`
4. **最终输出被 `_finalize_textgrid` 掩盖**：`_finalize_textgrid` 在所有 tier 上运行 `_NVV_PATTERN.sub` 包裹 `<>`，所以 hanzi tier 中 `LAUGHTER` 被重新包裹为 `<LAUGHTER>`

### 为什么之前未被发现

此 bug 被三层防护掩盖：
1. `is_word_like` 过滤掉裸 `<` `>` → 不在 ref_cjk/ref_alpha 中 → 不参与游标消费
2. `_alpha_text_matches` 内部 strip `<>` → `clean` 匹配仍成功 → cursor 推进
3. `_finalize_textgrid` 事后包裹 `<>` → 最终输出括号正确

但在以下场景可能暴露：
- 多个 NVV 相邻（如 `<LAUGHTER><BREATHING>`）→ `_extract_word_chars` 产生 `['<','LAUGHTER','>','<','BREATHING','>']` → 两个 `>` 被当作独立 punct → `is_word_like` 过滤 → ref_alpha = `['LAUGHTER','BREATHING']` → 游标消费仍正确但中间状态混乱

### 修改点

**AJ. `_extract_word_chars` — `<` `>` 特殊处理** (~line 1004-1032)

修改前：
```python
elif c.isalpha() or c == '-':
    buf += c
elif c.isdigit():
    buf += c
else:
    # punct / whitespace
```

修改后：
```python
elif c == '<':
    if buf:
        result.append(buf)
        buf = ""
    buf += c
elif c == '>':
    buf += c
    result.append(buf)
    buf = ""
elif c.isalpha() or c == '-':
    buf += c
elif c.isdigit():
    buf += c
else:
    # punct / whitespace
```

效果：
- `<LAUGHTER>` → `['<LAUGHTER>']` ✓
- `<LAUGHTER>你好` → `['<LAUGHTER>', '你', '好']` ✓（`>` flush 后下一个字独立）
- `<sp1>test` → `['<sp1>', 'test']` ✓（silence tag 也正确分组）
- `hello-world` → `['hello-world']` ✓（hyphen 不受影响）
- `a—b` → `['a', '—', 'b']` ✓（em-dash 仍正确拆分）

### 与其他 Case 的关系

| Case | 问题 | 与本 Case 的关系 |
|------|------|----------------|
| 13 | NVV 大小写不一致 | 同属 NVV 文本处理链：Case 13 修复识别，本 Case 修复拆分 |
| 14 | hanzi tier NVV 括号污染 | Case 14 修复了 alpha 分支的 `<>` strip，本 Case 从源头防止 `<>` 被拆分 |
| 10 | hanzi tier 拼音残留 | 同属 `_extract_word_chars` → `_build_hanzi_tier` 数据流 |

### 验证方法

```python
assert _extract_word_chars("<LAUGHTER>你好") == ["<LAUGHTER>", "你", "好"]
assert _extract_word_chars("<sp1>test") == ["<sp1>", "test"]
assert _extract_word_chars("<BREATHING>hello") == ["<BREATHING>", "hello"]
assert _extract_word_chars("hello-world") == ["hello-world"]  # 不受影响
assert _extract_word_chars("a—b") == ["a", "—", "b"]  # 不受影响
```

---

---
## Case 16: MFA `--fine_tune` 默认开启导致边界漂移 + 对齐失败

**日期**: 2026-07-28
**涉及文件**: `scripts/run_pipeline.py`
**涉及函数**: `step_mfa_align`, `DEFAULT_CFG`
**触发场景**: 所有含 NVV token、BGM、英文词、短虚词的音频（花礼/鸣海派/Xiaoyuan/嘉然等直播数据集）

### 现象

**修复前**（fine_tune 默认 True）:
- hualishaxuan 全部 4 次 MFA align 尝试均失败（exit_code=1）
- CTC 锚点经 `adjust_ctc_boundaries.py` 能量修正后质量良好，但 MFA fine_tune 将其浮动到错误位置
- `_snap_to_ctc` 无法纠正：fine_tune 偏移 ≤0.1s，远低于 snap 阈值 0.3s

**修复后**（fine_tune 默认 False）:
- MFA 对齐成功（已验证：无 fine_tune 的测试运行 exit_code=0）
- 词边界保持在 adjust_ctc_boundaries 修正后的 CTC 锚点位置
- MFA 仍能在固定词边界内做音素级对齐（首次对齐 pass 不受 fine_tune 影响）

### 根因链

1. **adjust_ctc_boundaries.py**: 用音频能量分析精细化 CTC 词边界 → 产出高质量锚点
2. **MFA `--fine_tune`**（默认 True，tolerance=0.1s）: 用纯语音训练的声学模型浮动锚点边界。模型对 NVV（无模型）、BGM（未训练）、短虚词（能量弱）、英文（错误模型）均表现差 → 锚点被推偏
3. **_snap_to_ctc**（snap_threshold=0.3s）: 检查 MFA 偏离 CTC >0.3s 时拉回。fine_tune 仅移动 ≤0.1s → 永远不会触发 snap → 漂移永久保留

**设计冲突本质**: `adjust_ctc_boundaries`（能量修正）和 MFA fine_tune（声学模型修正）对同一数据做出矛盾决策，而 _snap_to_ctc 的阈值设计使后者覆盖前者。

### 修改点

**A. `step_mfa_align` — fine_tune 默认值 True → False** (~line 1002)

```python
# 修改前
if mc.get("fine_tune", True):        # 默认开启
    mfa_args.append("--fine_tune")
fine_tune_tolerance = mc.get("fine_tune_boundary_tolerance", 0.1)  # 100ms

# 修改后
if mc.get("fine_tune", False):       # 默认关闭
    mfa_args.append("--fine_tune")
fine_tune_tolerance = mc.get("fine_tune_boundary_tolerance", 0.02) # 20ms
```

**B. `DEFAULT_CFG["mfa"]` — 新增默认值** (~line 205)

```python
"fine_tune": False,          # DISABLED: adjust_ctc_boundaries already refines anchors
"fine_tune_boundary_tolerance": 0.02,  # only used when fine_tune: true
```

**C. `config.yaml` — 参考文档同步更新**

### 关联样本

- hualishaxuan (花礼筛选): 87 文件全部 MFA 对齐失败
- FILTER_ANALYSIS_REPORT.md: ~60% word_in_silence 命中归因于 fine_tune
- 已禁用 fine_tune 的配置（均正常）: hechengria_test100, test_shayi_1, aed_3files, test_en_mfa, hechengria_10, test_hechengria_1

### 验证方法

```python
# 检查 command_history.yaml 中的所有 MFA align 调用
# 修复后不应再出现 --fine_tune 标志（除非数据集显式配置）
```

---

## Case 17: NVV token 被 `_normalize_word_spellings` 改写 + 首词为标点 + pinyin_phones 断层

**日期**: 2026-07-28
**涉及文件**: `scripts/postprocess_textgrids.py`, `scripts/ctc_prealign.py`
**涉及函数**: `_normalize_word_spellings`, `strip_edge_punctuation`, `build_pinyin_phones_tier`, `_normalize_punct`, `_sync_derived_tiers`
**触发样本**: 花礼_emo来听小鼠歌吧_20250504_2058_3700c4b4_clip0003_clip0012

### 现象

修复前：
```
words tier:       <sp1>[0.000-1.170]  …[1.170-1.710]  SURPRISE[1.710-1.950]  qie4[1.950-2.150]
pinyin_phones:    <sp1>[0.000-1.170]  …[1.170-1.710]  en:tɕʰ[1.940-1.950]  q[1.950-2.100]
```
三个症状：
1. **`…` 为首词**: 省略号出现在 `<sp1>` 之后，SURPRISE 之前
2. **SURPRISE pinyin_phones 断层**: 词区间 [1.71-1.95] 内 pinyin_phones 为 `en:tɕʰ`（这是下一词 `qie4` 的声母泄漏），SURPRISE 本体无音素覆盖，形成 230ms 空洞
3. **pinyin_phones 与 hanzi/words 不同步**: 三级轨道边界不重合

### 根因链

**子问题 A: `…` 为首词**

1. **NVASR CTC 预处理** (ctc_prealign.py): 原始 ASR 输出 `<|zh|><|HAPPY|>…[Surprise-oh]切片OP…`，NVASR 在产出 `_text_cn.txt` 时去掉 `<|zh|>` `<|HAPPY|>` 标签，但 `…` 残留在文本开头
2. **`_punct.json` 保留 `…`**: CTC 停顿检测将 `…` 作为标点条目写入 `_punct.json`，时间戳 [1.17-1.71]
3. **`_inject_punctuation`**: 将 `…` 插入 words tier，填充 `<sp1>` 与 SURPRISE 之间的间隙 → `…` 成为首词

**子问题 B: SURPRISE 音素断层**

4. **`_normalize_punct` 连字符→逗号**: `_text_cn.txt` 中 `SURPRISE-OH` 经过 `_normalize_punct` 时，`is_punct('-')` 返回 True 且 `-` 不在白名单中 → 替换为 `，` → `_text_cn.txt` 变成 `SURPRISE，OH`。同批次 18 个含连字符 NVV 的文件 100% 受影响。但 `.lab`/`_tokens.jsonl` 不被 `_normalize_punct` 修改 → 保持 `SURPRISE-OH`
5. **MFA align**: `SURPRISE-OH` 无对应声学模型 → MFA 将其对齐为 `spn` [1.72-1.94]
6. **`_normalize_word_spellings`** (Phase 5): 用 raw_text 中的 `SURPRISE`（`_finalise_textgrid` 吞掉了 `，OH` 标点）替换 words tier 中的 `surprise-oh`。改写后 SURPRISE 不再是 NVV token → `is_nvv_token("SURPRISE")` = False → 被当作英文词处理
7. **English MFA**: SURPRISE 不在 en_mfa_windows 中 → 无声学音素数据
8. **`build_pinyin_phones_tier`**: 收集 SURPRISE 词区间内的 MFA phones → `spn` 被过滤，只有 `tɕʰ` [1.94-1.95] 残留（从 qie4 泄漏）。`en_mfa_windows` 无 SURPRISE 条目时，使用 `(w_iv.xmin, w_iv.xmax)` 作为缺省窗口 → 泄漏音素未被过滤 → pinyin_phones 出现 `en:tɕʰ`（错误标注）
9. SURPRISE 的 pinyin_phones 覆盖范围 [1.94-1.95] 与词区间 [1.71-1.95] 不重合 → 230ms 空洞

**子问题 C: 轨道不同步（系统性问题）**

10. postprocess 处理链中，多个阶段独立修改 words tier（`_snap_to_ctc`、`_refine_boundaries_by_energy`、`_inject_punctuation`、Phase 4 合并操作），但 hanzi 和 pinyin_phones 的同步重建仅在 Phase 5 末尾进行
11. Phase 4 的 D2 (`absorb_nvv_trailing`)、D3 (`absorb_silence_into_punct`)、D4 (`strip_edge_punctuation`) 在 textgrid 上原地修改 words tier 边界，但不触发 hanzi/pinyin_phones 重建 → Phase 4 后续 E/F/G 步骤读取的 pp_tier 是过时数据
12. NVV token 一旦被 `_normalize_word_spellings` 改写为普通英文词，所有下游 NVV 处理逻辑（边界吸收、自引用音素）均被跳过

**架构根因**: 管线没有统一的 tier 同步机制。words tier 是权威数据源，hanzi 和 pinyin_phones 是派生数据，但派生数据的重建散布在各处（Phase 3B、3.5、5），缺乏系统性的"修改 words → 立即重建派生 tier"的不变式。

### 修改点

**A. `_normalize_word_spellings` — NVV token 保护** (~line 1374)

```python
# 修改前
if ref_spelling != w_text and ref_spelling.isascii():
    words_tier.intervals[wi].text = ref_spelling

# 修改后 — 增加 NVV 检查
if is_nvv_token(w_text):
    continue  # 永远不改写 NVV token 的文本
if ref_spelling != w_text and ref_spelling.isascii():
    words_tier.intervals[wi].text = ref_spelling
```

**B. `strip_edge_punctuation` — 新增函数** (~line 852)

新增函数扫描 words tier，将首词前/尾词后的孤立标点吸收到相邻区间。在 Phase 4 中 `absorb_silence_into_punct` 之后调用。

**C. `build_pinyin_phones_tier` — 泄漏音素过滤** (~line 469)

```python
# 当第一个非静音音素起点超过词起点 30% 时长时，
# 判定为相邻词泄漏音素，清空 word_phones 走自引用 fallback
```

**D. `build_pinyin_phones_tier` — en_mfa_windows 缺失时的安全 fallback** (~line 521)

```python
# 修改前: en_mfa_windows.get(wl, (w_iv.xmin, w_iv.xmax))
# → 缺省窗口 = 整个词区间 + 0.3s margin → 泄漏音素全部通过

# 修改后: 若 wl 不在 en_mfa_windows 中，清空 word_phones
# → 走自引用 fallback，整词区间覆盖
```

**E. `_normalize_punct` — NVV 标签内部连字符保护** (~line 651, `ctc_prealign.py`)

```python
# 修改前 — '-' 被 is_punct 判为标点，非白名单 → 替换为 '，'
for ch in text:
    if is_punct(ch):
        char_info.append(("punct", ch in _NORM_ALLOWED_PUNCT, ch))

# 修改后 — 字母间的 '-' 是 NVV 标签连字符，不当作标点
for i, ch in enumerate(text):
    is_hyphen_in_nvv = (
        ch == '-'
        and i > 0 and i + 1 < len(text)
        and text[i - 1].isascii() and text[i - 1].isalpha()
        and text[i + 1].isascii() and text[i + 1].isalpha()
    )
    if is_punct(ch) and not is_hyphen_in_nvv:
        char_info.append(("punct", ch in _NORM_ALLOWED_PUNCT, ch))
```

此修复从源头阻止了 `SURPRISE-OH` → `SURPRISE，OH` 的转换。同批次
18 个含连字符 NVV 的文件全部受益。配合修改 A（NVV 文本保护），
形成双重防护。

**F. `_sync_derived_tiers` — 统一 tier 同步函数** (~line 879, 新增)

新增 `_sync_derived_tiers(textgrid, ...)` 函数，从当前 words + phones tier
重建 hanzi 和 pinyin_phones。在 Phase 4 的两个关键点调用：

1. **Phase 4 D4 之后**: D2/D3/D4 原地修改 words tier → 立即同步，确保后续 E/F/G
   读取的 pp_tier 是新鲜的
2. **Phase 4 末尾**: 确保进入 Phase 5 前所有派生 tier 与 words tier 一致

```python
def _sync_derived_tiers(textgrid, ipa_to_pinyin, pinyin_dict,
                         raw_text, en_mfa_windows, report_warnings):
    # 1. Rebuild hanzi from updated words tier
    # 2. Rebuild pinyin_phones from updated phones + words tiers
```

**架构原则**: words tier 是唯一权威数据源。hanzi 和 pinyin_phones 是派生 tier。
任何修改 words tier 边界/文本的操作，必须立即通过 `_sync_derived_tiers` 重建
派生 tier，不允许延迟到 Phase 5。"修改谁，同步全部"。

### 关联样本

- 花礼_emo来听小鼠歌吧_20250504_2058_3700c4b4_clip0003_clip0012: 触发全部三个子问题
- 此前 Cases Q, R, S, T (2026-07-27) 已部分修复轨道同步，但未覆盖 NVV 文本改写 + 泄漏音素场景

### 验证方法

```python
# 验证 NVV 保护
assert is_nvv_token("SURPRISE-OH")  # True — NVV 名称完整保留
assert not is_nvv_token("SURPRISE") # False — 被截断的形式不会被误判

# 验证边缘标点剥离
# 任何文件的 words tier 首 interval (非 silence/NVV) 不应是标点
# 任何文件的 words tier 尾 interval (非 silence) 不应是标点

# 验证音素无泄漏
# pinyin_phones 中 NVV token 的词区间内不应出现以 en: 前缀的音素
# (除非该词确实在 en_mfa_windows 中有条目)

# 验证 _normalize_punct 不再破坏 NVV 连字符
# 含连字符的 NVV (SURPRISE-OH, QUESTION-YI 等) 在 _text_cn.txt 中应保留 '-'
# 不应出现 SURPRISE，OH 形式的逗号拆分
```

---

## Case 18: NVASR 情绪标签后残留开头标点 → hanzi 首词为 `，`

**日期**: 2026-07-29
**涉及文件**: `scripts/ctc_prealign.py`
**触发样本**: 花礼_八分音符鼠_20241122_2201_d8293668_clip0000_clip0000

### 现象

```
ASR 输出:   <|zh|><|SAD|>， call恩你...
去除标签:                 ， call恩你...   ← 逗号成为第一个字符
hanzi:      ， | call | 恩 | ...           ← 首 interval 是标点
```

### 根因链

1. SenseVoice 在音频开头检测到情绪标签（`<|SAD|>`）后自动附加标点
2. `text_cn` 清洗只去标签不移除紧随其后的标点
3. CTC 为该标点分配 0s 锚点 → postprocess 注入 → hanzi 首词是标点

### 修改点

**A. `ctc_prealign.py` — 输出前检测并删除开头标点** (~L1196, ~L1277)

分两处：part A 从 `punct_entries` 删除首个标点锚点（start<100ms），part B 从 `text_asr` 中删除标点字符。后续词时间戳不平移。


## Case 19: `_snap_to_ctc` 丢弃开头静音 → hanzi 缺失 `<sp1>` interval

**日期**: 2026-07-29
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_snap_to_ctc`
**触发样本**: 花礼_八分音符鼠_20241122_2201_d8293668_clip0000_clip0000

### 现象

```
MFA raw:   <eps>[0~1.88]  call[1.88~...]     ← 有开头静音
最终输出:  call[1.95~...]                      ← 开头静音消失，raw_text 有 <sp1> 但 words/hanzi 无对应 interval
```

### 根因链

1. `_snap_to_ctc` 的 `mfa_words` 过滤掉所有 silence/`<eps>`（L3284），只处理内容词
2. 重建 words tier 时只处理了尾静音（L3610-3614），未补回开头静音
3. 第一个词起点前的 gap 没被填充 → 开头静音 interval 丢失

### 修改点

**A. `_snap_to_ctc` — 补回头部静音** (~L3610)

尾静音处理前插入：首个 word start>0.005s 时，在 `new_word_ivs` 头部补入 silence gap。

**B. `_finalize_textgrid` — 兜底插入** (~L148)

多 interval tier 首 interval 不从 0 开始且非 `<spN>` 时，补插 `<sp1>(0, first.xmin)`。确保即便上游丢弃了静音 interval，最终输出仍有 `<sp1>`。


## Case 20: strip_edge_punctuation 尾随标点过度剥离 → 句尾 。！？ 全部丢失

**日期**: 2026-07-29
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `strip_edge_punctuation`, `process_one`
**触发样本**: 花礼_不困小鼠_20241117_2158_4ad6dee3_clip0023_clip0001

### 现象

所有 pipeline 输出的句尾标点（`。！？`）全部消失：

```
_text_cn.txt:     ...知道什么意思吧？            ← 有 ？
TextGrid raw_text: ...知道什么意思吧              ← ？ 消失
TextGrid words:    ... ba5[10.22-10.51] <sp2>...  ← 无 ？ interval
TextGrid hanzi:    ... 吧                          ← 无 ？
```

受影响的 87 个文件中，几乎所有以 `。！？` 结尾的句子都丢失了结尾标点。

### 设计缺陷："镜像处理"的逻辑谬误

`strip_edge_punctuation` 是在 Case 17-B 中新增的函数，其设计意图是剥离**孤儿标点**——
NVASR 去除情绪标签（`<|HAPPY|>`、`<|zh|>` 等）后，标签之间的标点（主要是 `…`）
残留在文本开头/结尾，被 CTC 分配时间锚点后注入 words tier，成为孤立的首/尾标点 interval。

函数的处理逻辑是对称的：

```
for i in range(0, first_real):         ← 开头: 第一个实词前的 punct → 剥离
    if is_punct(intervals[i].text): ...
for i in range(last_real + 1, len):    ← 尾随: 最后一个实词后的 punct → 剥离
    if is_punct(intervals[i].text): ...
```

**开头剥离是正确的**：在第一个实词之前，不应该存在任何标点。
如果出现（情绪标签剥离残留），它必然是孤儿，应该被吸收。

**尾随剥离是错误的**：在最后一个实词之后，标点分两种：
1. **孤儿 `…`**：NVV 标签剥离残留 → 但情绪标签几乎只在句首，不在句尾。即使标签在句尾（极其罕见），`你好<|HAPPY|>…` 去标签后变 `你好…`——这与正常句尾省略号**无法区分**，也不需要剥离
2. **合法句尾标点 `。！？…`**：ASR 原始文本的句末标点 → **必须保留**

尾随剥离从 Case 17-B 引入之初就没有需要解决的实际 bug——它是作为开头剥离的"镜像"顺手加上的。开头剥离有明确的孤儿标点场景（情绪标签在句首），尾随剥离没有。两者不是对称场景。

**为什么"镜像"在此不成立**：

句子结构是 `[内容词...] [标点]`，标点天然在句尾。开头的标点一定是异常的，
但尾随的标点大多数是合法的。开头和尾随不是对称场景，镜像处理是设计错误。

### 根因链

1. **Case 17-B 引入 `strip_edge_punctuation`** (~L938)：解决 NVASR 剥离 `<|HAPPY|>` 情绪标签后残留 `…` 成为首词的问题。设计为"镜像处理"——开头的孤儿标点和尾随的标点一起剥离
2. **镜像假设错误**：句子结构是 `[内容] [标点]`，标点天然在末尾。开头标点必然是异常，尾随标点大多数是合法的。两者不是对称场景
3. **尾随剥离用 `is_punct()` 一刀切** (~L997-1011)：从 `last_real`（最后一个实词）之后扫描所有 interval，凡是 `is_punct()` 的一律清空，无法区分孤儿 `…` 和合法句尾标点 `。！？`
4. **absorber 链已覆盖尾随清理**：`absorb_nvv_trailing` (Case 9 W1, Phase 4-D2) 和 `absorb_silence_into_punct` (Case 9 W2, Phase 4-D3) 在 `strip_edge_punctuation` (Phase 4-D4) **之前**执行，已经处理了 NVV 后的标点+静音链。尾随剥离不仅是逻辑错误的，而且是**冗余的**
5. **Phase 5 不可逆传播** (~L4654)：`raw_text` 从 `hanzi` tier 重建覆盖 → 被剥离的标点永久消失，无法恢复

### 完整数据流追踪

以 `花礼_不困小鼠_clip0023` 为例，追踪 `？` 的完整生命周期：

```
Phase 1-2:  _text_cn.txt → raw_text 变量 → raw_text tier
            "...知道什么意思吧？"                       ✅ ？ 存在

Phase 3C:   _inject_punctuation: CTC ？[11.07-11.16] 注入 words tier
            words: ... ba5[10.22-10.51] <sp2>[10.51-11.125] ？[11.125-11.16]
            last_punct → punct_start=ba5.xmax=10.51
            ？ → (10.51, 11.125, "？", "punct")  [615ms]  ✅ ？ 存在

Phase 4-D2: absorb_nvv_trailing — 无 NVV → 无操作

Phase 4-D3: absorb_silence_into_punct — 无残余 <spN> → 无操作

Phase 4-D4: strip_edge_punctuation  ← 问题发生点！
            last_real = ba5 的 index
            for i in range(last_real+1, len):    # 扫描 ba5 之后
              <sp2> → is_punct? NO  → skip
              ？   → is_punct? YES → trailing_punct_indices.append(i)
            → ？ 被清空: text="", xmin=0, xmax=0
            → <sp2>.xmax 扩展到 11.16 (吸收 ？ 的时间)
                                               ❌ ？ 被剥离！

Phase 5:    _build_hanzi_tier(words) → hanzi 无 ？
            raw_text = "".join(hanzi) → 覆盖 raw_text tier
                                               ❌ ？ 永久消失！
```

### 触发条件

任何满足以下条件的句子：
- 最后一个实词后存在标点（`。！？`，经 `_inject_punctuation` 注入 words tier）
- `strip_edge_punctuation` 将该标点识别为 "trailing punct" → 清空

### 修改点

**A. `strip_edge_punctuation` — 删除尾随标点剥离逻辑** (~L997-1017)

尾随剥离从 Case 17-B 引入之初就是错误的——它作为开头剥离的"镜像"被顺手加上，
但没有任何实际 bug 驱动。情绪标签几乎只在句首（`<|HAPPY|>…内容`），不在句尾。
句尾的标点都是合法的句末标点，不应被剥离。

修改前 (~L997-1011, 与开头剥离对称的镜像代码):
```python
# ── Strip trailing punct: absorb into the following interval ──
trailing_punct_indices = []
for i in range(last_real + 1, len(intervals)):
    if is_punct(intervals[i].text):
        trailing_punct_indices.append(i)

for pi in sorted(trailing_punct_indices, reverse=True):
    p_iv = intervals[pi]
    if pi + 1 < len(intervals):
        intervals[pi + 1] = _replace(intervals[pi + 1], xmin=p_iv.xmin)
    elif pi > 0:
        intervals[pi - 1] = _replace(intervals[pi - 1], xmax=p_iv.xmax)
    intervals[pi] = _replace(intervals[pi], xmin=0, xmax=0, text="")
```

修改后（整个尾随剥离代码块删除，替换为注释说明）:
```python
    # NOTE: There is intentionally NO trailing strip.
    # Trailing punctuation (。！？…) after the last real word is ALWAYS
    # legitimate — sentences naturally end with punctuation.  The "mirror"
    # design (stripping both edges) is a logical error because leading
    # and trailing edges are NOT symmetric:
    #   - Leading punct: always orphaned (tag-stripping artifact) → strip
    #   - Trailing punct: always legitimate (end-of-sentence) → keep
    # NVV-trailing punct+silence chains are already handled upstream by
    # absorb_nvv_trailing (Case 9 W1) and absorb_silence_into_punct (Case 9 W2).
```

函数现在只做一件事：剥离开头（first_real 之前的）孤儿标点。这是它唯一的实际职责。

**B. `_inject_punctuation` — 最后标点 30ms 地板保护** (~L2482-2484, 防御性)

修改前：
```python
new_merged.append((punct_start, words_tier.xmax, punct_text, "punct"))
```

修改后：
```python
punct_end = max(punct_start + 0.030, words_tier.xmax)
new_merged.append((punct_start, punct_end, punct_text, "punct"))
```

防止末词 xmax == words_tier.xmax 时标点坍缩为零时长（独立于本 Case 的边缘保护）。

### 与关联 Case 的关系

| Case | 问题 | 与本 Case 的关系 |
|------|------|----------------|
| 9 (W1/W2) | NVV后标点+静音链未被吸收 | `absorb_nvv_trailing` + `absorb_silence_into_punct` 已覆盖，尾随剥离冗余 |
| 17-B | `…` 为首词（NVV 标签剥离残留） | 本 Case 的根因——17-B 引入 `strip_edge_punctuation`，尾随剥离过于激进 |
| 18 | 情绪标签后残留开头标点 | ctc_prealign 层面修复，不依赖 postprocess 剥离 |

### Phase 4 执行顺序与责任分工

```
Phase 4 执行顺序:
  D.  handle_unexpected_silences    ← gap 级 <sp0> 无条件合并 (Case 8)
  D2. absorb_nvv_trailing           ← NVV 吞并尾部标点+静音链 (Case 9 W1)
  D3. absorb_silence_into_punct     ← 兜底: 标点吸收残余 <spN> (Case 9 W2)
  D4. strip_edge_punctuation        ← 边缘标点剥离 (Case 17-B)  ← 本 Case
  ── SYNC ── _sync_derived_tiers   ← words → hanzi + pp 同步 (Case 17-F)
```

D2 已经处理了 "NVV → 标点 → 静音" 链（如 `<LAUGHTER> → ！ → <sp2>`），
D3 兜底处理了 "标点 → 残余 `<spN>`"。到 D4 执行时，所有 NVV 相关的尾随孤儿标点
已经被 D2 吸收。D4 扫到的尾随标点**只可能是合法的句末标点**。

D4 的尾随剥离不仅逻辑错误（镜像假设不成立），而且在执行顺序上也是冗余的。

### 验证方法

```python
# 句尾标点必须在 words tier 中存在
for iv in words_tier.intervals:
    if iv.text.strip() in '。！？':
        assert iv.xmax > iv.xmin + 0.001, f"zero-duration ending punct: {iv.text}"

# raw_text tier 结尾标点应与 _text_cn.txt 一致
assert raw_text_raw.endswith('？') == text_cn_raw.endswith('？')
```

### 关联样本

- `花礼_不困小鼠_20241117_2158_4ad6dee3_clip0023_clip0001`：`知道什么意思吧？` → `？` 丢失
- 花礼筛选全部 87 文件：句尾 `。！？` 系统性丢失


## Case 21: `is_punct` 误判 `<spN>` 为标点 → `strip_edge_punctuation` 开头静音被吸收进首词

**日期**: 2026-07-29
**涉及文件**: `scripts/postprocess_textgrids.py`, `scripts/pipeline_utils.py`
**涉及函数**: `strip_edge_punctuation`, `is_punct`
**触发样本**: 花礼_变身_小鼠天使_20240924_2105_a90ca312_clip0010_clip0001

### 现象

首词 `喵` 从 0s 开始，开头静音 `<sp2>` 完全消失：

```
修复前:
  words:  miao1[0.000-1.830]  …[1.830-2.010]  hao3[2.010-...]
  hanzi:  喵[0.000-1.830]      …[1.830-2.010]  好[2.010-...]
          ↑ 开头静音消失，首词从 0s 开始

修复后:
  words:  <sp2>[0.000-1.410]  miao1[1.410-1.830]  …[1.830-2.010]  hao3[2.010-...]
  hanzi:  <sp2>[0.000-1.410]  喵[1.410-1.830]      …[1.830-2.010]  好[2.010-...]
          ↑ 开头静音正确保留
```

MFA 对齐中 `<eps>` [0.000-1.940] 正确存在于开头，但最终输出中消失。

### 根因链

1. **`is_punct()` 定义过于宽泛** (`pipeline_utils.py` L771-773)：
   ```python
   def is_punct(s: str) -> bool:
       return bool(s.strip()) and not is_word_like(s)
   ```
   `<sp2>` 以 `<` 开头，不是 alpha/CJK/digit，所以 `is_word_like("<sp2>")` → False，进而 `is_punct("<sp2>")` → True。

2. **`strip_edge_punctuation` 开头剥离未排除静音** (L983-985)：开头剥离扫描 `first_real` 之前的所有 interval，发现 `<sp2>` → `is_punct` True → 加入剥离列表

3. **剥离逻辑将静音吸收进首词** (L987-995)：`<sp2>` 是第一个 interval（pi=0），无前序 interval → `elif pi+1 < len`：下一个 interval (`miao1`) 的 xmin 被设为 `<sp2>.xmin = 0.0`

### 与 Case 20 的同源性

Case 20 和 Case 21 的根因都指向同一个设计问题：

| | Case 20（尾随） | Case 21（开头） |
|------|------|------|
| 被剥离对象 | `。！？`（合法句尾标点） | `<sp2>`（开头静音标签） |
| `is_punct` 返回值 | True（正确——确实是标点） | True（**错误**——静音不是标点） |
| 问题本质 | 设计错误：不应剥离尾随 | **bug**：`is_punct` 定义太宽，未排除 `<spN>` |

两者都在 `strip_edge_punctuation` 中触发，但性质不同。Case 20 已经通过删除尾随剥离解决，Case 21 需要修复开头剥离中的 `is_silence` 前置检查。

### 修改点

**A. `strip_edge_punctuation` — 开头剥离增加 `is_silence` 检查** (~L983-985)

修改前：
```python
leading_punct_indices = []
for i in range(first_real):
    if is_punct(intervals[i].text):
        leading_punct_indices.append(i)
```

修改后：
```python
leading_punct_indices = []
for i in range(first_real):
    if not is_silence(intervals[i].text) and is_punct(intervals[i].text):
        leading_punct_indices.append(i)
```

`<spN>` 静音标签在 `is_punct` 检查之前被 `is_silence` 排除，不会被误剥离。

### Case 20 + 21 的综合修复

`strip_edge_punctuation` 函数最终的职责范围：

```
strip_edge_punctuation:
  ├─ 开头剥离: first_real 之前的 punct (排除 <spN>) → 吸收   ← Case 21 修复
  └─ 尾随剥离: 已删除                                       ← Case 20 修复
```

这两个修复一起，使函数只做它最初设计要做的一件事：剥离情绪标签去除后残留在句首的孤儿 `…`。

### 关联样本

- `花礼_变身_小鼠天使_20240924_2105_a90ca312_clip0010_clip0001`：`<sp2>` 被吃 → 喵从 0s 开始
- 花礼筛选全部 87 文件：所有句首静音均受影响

### 潜在其他影响范围

`is_punct` 的宽泛定义可能在其他地方造成类似问题。任何仅用 `is_punct` 而未先检查 `is_silence` 的地方，都可能将 `<spN>` 误判为标点。搜索代码库中所有 `is_punct()` 调用可审计。


## Case 22: MFA 对齐偏差 → snap 修正后标点被 `<spN>` 替换 → 被吞标点恢复

**日期**: 2026-07-29
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (Phase 5 末尾)
**触发样本**: VR研学_小鼠粽子喜欢甜的还是咸的_20250531_2222_b330ee29_clip0010_clip0007, 3.15s 处的逗号

### 现象

ASR 识别出标点，CTC 有标点锚点，`_text_cn.txt` 中有标点，但最终 words tier 中标点消失，变成了 `<spN>` 静音间隙：

```
_text_cn.txt:   "...迪士尼，谢谢..."       ← ASR 有逗号
CTC punct:       ，[3.245-4.710]           ← 有锚点
最终 words:      ni2[2.75-3.15] <sp2>[3.15-3.81] xie4[3.81-4.05]
                 ↑ 逗号消失，变成 <sp2>
```

### 根因链

这是一个多步骤连锁反应：

1. **MFA 对齐偏差**：MFA 将 `xie4` 错放到 [3.15-3.26]（仅 110ms，实际应该是 [3.81-4.05]），导致 `ni2` 和 `xie4` 之间出现本不应存在的大段 `<eps>`

2. **`_snap_to_ctc` 修正词边界**：检测到 CTC 时长 > MFA 时长 ×2，把 `xie4` 拉到 CTC 正确位置 [3.81-4.05]。`ni2` [2.75-3.15] 和 `xie4` [3.81-4.05] 之间出现 660ms 间隙 → 被 `<sp2>` 填充。**词边界修正了，但间隙中的标点被埋没了**

3. **`_inject_punctuation` 重叠裁剪跳过静音**：`，` [3.245-4.710] 与 `<sp2>` [3.15-3.81] 重叠，但 `<sp2>` 因 `is_silence=True` 被重叠裁剪逻辑跳过（L2338-2340），逗号只与后续 `xie4` 发生裁剪，被推到 [4.29-4.71]

4. **逗号被孤立**：推到 [4.29-4.71] 的逗号位于 `xie4` 和 `shou1` 之间，但后续 Phase 4 步骤中可能被某个操作误吸收（具体步骤待精确追踪确认）。最终 `shou1` 从 CTC [4.71-4.95] 变成 [4.29-4.95]——逗号的时间被吸收了

5. **间隙变回 `<spN>`**：逗号消失后，`xie4` 和 `shou1` 之间的间隙被重新标记为静音

### 设计认知

这个 bug 的根源不是某个步骤的代码错误，而是**两种修正逻辑的交互冲突**：

| 步骤 | 做什么 | 副作用 |
|------|--------|--------|
| `_snap_to_ctc` | 把词边界拉到 CTC 锚点（修正 MFA 偏差） | 词之间产生新的间隙，埋没原本覆盖在上的标点 |
| `_inject_punctuation` | 把 CTC 标点注入 words tier | 重叠裁剪时 `<spN>` 不参与，标点被推向词边界之外 |

两者各自正确，但交互产生了一个"无主之地"——MFA 对齐偏差导致的间隙中，标点被推向边界后孤立，最终被后续 absorb 步骤吃掉。

### 修复策略

**事后兜底**（不改变现有逻辑，在 Phase 5 末尾做最后检查）：

在所有处理完成后，扫描 words tier 中的 `<spN>` 区间。若 CTC 标点锚点落在该区间内，且该标点未出现在 words tier 的其他位置，则直接将 `<spN>` 替换为该标点字符，时长不变。然后同步重建 hanzi、pinyin_phones、raw_text。

选择"事后兜底"而非"修改前置逻辑"的原因：
- `_snap_to_ctc` 的边界修正逻辑已经过 19 个 Case 的验证，不宜改动
- `_inject_punctuation` 跳过 `<spN>` 是刻意的（静音不参与标点裁剪），改动风险大
- 兜底修复只影响"标点被吞"这一种已确认的失效模式，不改变正常路径

### 修改点

**A. `process_one` Phase 5 — 被吞标点恢复** (~L4665 之后，QC 之前)

```python
# ── 被吞标点恢复 ──
# 若 <spN> 区间包含 CTC 标点锚点, 且该标点未在 words 中,
# 则替换 <spN> 为该标点, 时长不变, 同步所有派生 tier.
if punct_entries:
    _words_t = tier_by_name(new_tg, "words")
    if _words_t:
        _existing_punct = {iv.text.strip() for iv in _words_t.intervals
                           if is_punct(iv.text) and not is_silence(iv.text)}
        for iv in _words_t.intervals:
            if not is_silence(iv.text): continue
            for p in punct_entries:
                if p["word"] not in '，。！？': continue
                if p["word"] in _existing_punct: continue
                overlap = min(iv.xmax, p["end_s"]) - max(iv.xmin, p["start_s"])
                if overlap > 0:
                    iv.text = p["word"]
                    _existing_punct.add(p["word"])
                    # → 同步重建 hanzi, pinyin_phones, raw_text
                    break
```

### 与关联 Case 的关系

| Case | 问题 | 与本 Case 的关系 |
|------|------|----------------|
| 1-3 | MFA 尾静音被 snap 回词 | 同属 snap→间隙→标点问题的前置条件 |
| 7 (Mod T) | ≤5ms 间隙吸收 | snap 后间隙处理的同族逻辑 |
| 12 | overlap fix 产生倒置 interval | snap 副作用导致标点异常 |
| 20 | strip_edge_punctuation 尾随剥离 | 另一个导致标点消失的路径（已修复） |

### 验证方法

```python
# 所有 CTC 标点锚点必须在 words tier 中有对应
for p in punct_entries:
    if p["word"] in '，。！？':
        found = any(iv.text.strip() == p["word"] and
                    abs(iv.xmin - p["start_s"]) < 0.5
                    for iv in words_tier.intervals)
        assert found, f"CTC punct {p['word']} at {p['start_s']}s not in words tier"
```

### 关联样本

- `VR研学_小鼠粽子喜欢甜的还是咸的_20250531_2222_b330ee29_clip0010_clip0007`：3.15s 逗号被吞

### 被吞标点恢复的触发条件（精确版）

```
① CTC 标点锚点存在于 _punct.json（ASR 识别到了）
② _inject_punctuation 后该标点在 words tier 中（快照确认）
③ Phase 4 后该标点从 words tier 消失（比对确认 → _swallowed_puncts）
④ 该位置现在是 <spN>（间隙被重新暴露）
→ 四个条件全部满足 → 替换 <spN> 为原标点
```

只恢复"确认被吞"的标点，不会误恢复从未存在过的标点。


## Case 23: `step_normalize_punct` 缺少 NVV 连字符保护 → `QUESTION-EI` 被拆成 `QUESTION，EI`

**日期**: 2026-07-29
**涉及文件**: `scripts/run_pipeline.py`
**涉及函数**: `step_normalize_punct`
**触发样本**: 花礼_和帕小聊_20241221_0914_7103ddf1_clip0016_clip0008, 6.99s 处的 `QUESTION-EI`

### 现象

含连字符的 NVV token `QUESTION-EI` 在 pipeline 处理后被拆散：

```
修复前:
  words:         <QUESTION-EI>[6.99-7.29]  EI[7.29-7.38]        ← EI 独立成词
  hanzi:         QUESTION[6.99-7.29]       EI[7.29-7.38]        ← 无 <>，被拆开
  pinyin_phones: <<QUESTION-EI>>[6.99-7.29]  en:ei[7.29-7.38]   ← 双重包裹 + 音素泄漏

修复后:
  words:         <QUESTION-EI>[6.99-7.29]                       ← 完整 NVV
  hanzi:         QUESTION-EI[6.99-7.29]                         ← 正确
  pinyin_phones: <QUESTION-EI>[6.99-7.29]                       ← 正确
```

### 根因链

1. **NVASR 输出** `[Question-ei]`（NVV 方括号格式），`ctc_prealign` 转换为 `QUESTION-EI`
2. **`.lab` 文件**中 `QUESTION-EI` 保留连字符（不经过 `step_normalize_punct`）
3. **`_text_cn.txt`** 经过 `step_normalize_punct`（Phase 2 标点分类）：
   ```python
   for ch in text:
       if is_punct(ch):  # '-' 命中! is_punct 定义为 not is_word_like
           char_info.append(("punct", ch in ALLOWED_PUNCT, ch))
           # '-' 不在白名单 → 后续被替换为 ，
   ```
4. **`_text_cn.txt` 变成** `QUESTION，EI`（连字符变逗号）
5. **MFA 对齐**：`.lab` 中有 `QUESTION-EI EI`（lab 不受影响），MFA 将 `EI` 当作独立词
6. **Postprocess**：`EI` 被 `is_english_token` 识别 → 送入 English MFA → 泄漏 `en:ei` 音素
7. **Hanzi tier**：`QUESTION` 不是 NVV 格式（无 `<>`），`_build_hanzi_tier` 将其当作普通英文词

### Case 17-E 修复不完整

Case 17-E 在 `ctc_prealign._normalize_punct` 中添加了 NVV 连字符保护，但 **`run_pipeline.step_normalize_punct` 没有被同步修复**。两个函数做同样的标点规范化，但只有前者有 `is_hyphen_in_nvv` 检查。

```
ctc_prealign._normalize_punct     ← Case 17-E 已修复 ✅
run_pipeline.step_normalize_punct ← 遗漏！ ❌ → Case 23
```

### 修改点

**A. `step_normalize_punct` — Phase 2 增加 `is_hyphen_in_nvv` 检查** (~L680)

修改前：
```python
for ch in text:
    if is_punct(ch):
        char_info.append(("punct", ch in ALLOWED_PUNCT, ch))
    else:
        char_info.append(("other", None, ch))
```

修改后：
```python
for i, ch in enumerate(text):
    is_hyphen_in_nvv = (
        ch == '-'
        and i > 0 and i + 1 < len(text)
        and text[i - 1].isascii() and text[i - 1].isalpha()
        and text[i + 1].isascii() and text[i + 1].isalpha()
    )
    if is_punct(ch) and not is_hyphen_in_nvv:
        char_info.append(("punct", ch in ALLOWED_PUNCT, ch))
    else:
        char_info.append(("other", None, ch))
```

### 全面审计

对 pipeline 中所有处理 `_text_cn.txt` 标点规范化的函数进行了审计：

| 函数 | 文件 | 连字符保护 | 状态 |
|------|------|-----------|------|
| `_normalize_punct` | ctc_prealign.py | ✅ Case 17-E | 已修复 |
| `step_normalize_punct` | run_pipeline.py | ✅ Case 23 | **本次修复** |
| `normalize_punct_inline` | pipeline_utils.py | N/A（`-` 不在其处理范围） | 安全 |

### 关联样本

- `花礼_和帕小聊_20241221_0914_7103ddf1_clip0016_clip0008`：`QUESTION-EI` → `QUESTION，EI` + `EI` 独立成词


## 模板 (新 Case 用)

```markdown
## Case N: [标题]

**日期**: YYYY-MM-DD
**涉及文件**: `scripts/xxx.py`
**涉及函数**: `xxx`, `yyy`
**触发样本**: xxx

### 现象

[修复前 vs 修复后的数据对比]

### 根因链

1. [步骤1]: ...
2. [步骤2]: ...

### 修改点

**X. `xxx` — 修改描述** (~line N)

[代码 diff]

### 关联样本

- [样本]
```
