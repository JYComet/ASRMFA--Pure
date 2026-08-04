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
| 25 | 2026-07-29 | postprocess_textgrids.py | _inject_punctuation 重叠判断 + _punct_before 范围 + 恢复匹配策略 |
| 26 | 2026-08-04 | postprocess_textgrids.py | pinyin_phones 声母独占→韵母消失 (ch/zh/sh → 缺 final) |
| 27 | 2026-08-04 | postprocess_textgrids.py | words tier 区间重叠 — 相邻词时间边界交叉 |
| 28 | 2026-08-04 | postprocess_textgrids.py | 倒置 interval — xmin > xmax |
| 29 | 2026-08-04 | postprocess_textgrids.py | 极短内容词 (< 30ms) 物理不可能 |
| 30 | 2026-08-04 | postprocess_textgrids.py, adjust_ctc_boundaries.py, pipeline_utils.py | 参考文本模糊子串匹配 — 纯英文 CTC 锚点标定风险分析 |
| 31 | 2026-08-04 | normalize_english_tokens.py, ctc_prealign.py | 英文 CTC 锚点三重修复 — 碎片化 / 合并错误 / NVV 误判 |

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
| AQ | `ctc_prealign.py` ~1193,~1217 | **Case 24**: TextGrid 写入移至空检测之后，避免 ASR 空输出留下孤立的空 TextGrid |
| AR | `_inject_punctuation` ~2346 | **Case 25-A**: else 分支改为 split: word 在 punct 内时不删除, 拆分 punct 为左右两段 |
| AS | `postprocess_textgrids.py` ~4361,~4701 | **Case 25-B**: `_punct_before` 监控扩展至 `，。！？…、；：` |
| AT | `postprocess_textgrids.py` ~4717-4748 | **Case 25-C**: 吞标点恢复从时间重叠匹配改为 CTC 序列顺序匹配（前后邻词定位） |
| AU | `run_pipeline.py` ~238, `postprocess_textgrids.py` ~4942 | **Case 25-D**: bgm_threshold 天花板 `bgm_max_threshold` (默认 0.05)，防 60 分位噪声底被污染 |
| AV | `postprocess_textgrids.py` ~4961 | **Case 25-E**: bgm 检测从整段平均值改为 50ms 帧级别，≥20% 帧超阈值才触发 |
| AW | `postprocess_textgrids.py` ~4819-4888 | **Case 25-F**: 末尾标点强制保留：从前词截取 ≥60ms（弹性取 max(60ms, CTC原始时长)），同步 words/hanzi/pinyin_phones |
| AX | `postprocess_textgrids.py` ~3028-3051 | **Case 25-G**: `_refine_boundaries_by_energy` 标点边界保护：标点起点与词尾重叠 <100ms 且主体在词尾之后 → 阻止 dead silence 延伸 |
| AY | `run_pipeline.py` ~1049-1059, `align_english_mfa.py` ~382-393 | **Case 25-H**: 英文 MFA 词典/G2P 路径修复：pretrained_models fallback 到 PROJECT_ROOT.parent + G2P 失败时写 base dict 兜底 |
| AZ | `postprocess_textgrids.py` ~87-91 | **Case 25-I**: `_NVV_PATTERN` lookbehind/ahead 加 `<>` 排斥，防已包裹 NVV token 被 `_finalise_textgrid` 再次包裹 → pinyin_phones 出现 `<<TOKEN>>` |
| BA | `build_pinyin_phones_tier` ~510-527 | **Case 26-A**: dict 查询前移 + `word_phones` 空时按比例拆分 |
| BB | `build_pinyin_phones_tier` ~600-616 | **Case 26-B**: 三分支替代 `len(dict_phones)==1 or len(word_phones)<=1` |
| BC | `_INIT_FRAC` ~434 | **Case 26-C**: 按声母类型细化拆分比例 (塞音0.20/鼻边0.22/擦音0.28/塞擦0.35) |
| BD | `process_one` QC 段 | **Case 26-D**: 质检过滤 `init_only_phone` — pinyin_phones 仍残留纯声母 |
| CA | `_fix_overlapping_boundaries` (新增) | **Case 27-A**: 轻度边界重叠修复 (内容词split_overlap / punct clip_punct) |
| CB | `process_one` QC 段 | **Case 27-B**: 质检过滤 `overlapping_words` — 修复后仍重叠的进入 filtered/ |
| DA | `process_one` QC 段 | **Case 28**: 质检过滤 `inverted_interval` — xmin > xmax 倒置检测 |
| EA | `process_one` QC 段 | **Case 29**: 质检过滤 `short_word` — 内容词 < 30ms 检测 |

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


## Case 24: ASR 空输出先写 TextGrid 再检测 → 留下孤立的空 TextGrid

**日期**: 2026-07-29
**涉及文件**: `scripts/ctc_prealign.py`
**涉及函数**: `main` (推理循环 ~L1193-1218)
**触发样本**: `花礼_-晚间吃口小鼠_20251120_2059_365e41bc_clip0017_clip0019` (raw_text=`<sp1>`，纯静音)

### 现象

ASR 输出为空的 stem 在 `ctc_pretg/` 中留下孤立的空 TextGrid，但没有 `.lab`、`_tokens.jsonl`、`_punct.json`、`_text_cn.txt`、`_text_raw.txt`：

```
ctc_pretg/
  ├─ 花礼_-晚间吃口...TextGrid   ← 孤立的空 TG（words tier 仅一个空 interval）
  ├─ （没有 .lab）
  ├─ （没有 _tokens.jsonl）
  ├─ （没有 _punct.json）
  ├─ （没有 _text_cn.txt）
  └─ （没有 _text_raw.txt）
```

### 根因链

1. **写 TextGrid 在空检测之前**（L1193-1194）：无论 ASR 是否有输出，`write_textgrid()` 都在 `continue` 之前执行
2. **空检测在 L1213-1218**：`lab_tokens.strip()` 为空时打印 SKIP 并 `continue`，跳过 `.lab` 等后续文件写入
3. **下游步骤通过不同文件类型发现 stem**：`adjust_ctc` glob `*_tokens.jsonl`，`normalize_*` glob `*_text_cn.txt` / `*.lab`，MFA align 用 `.lab` 作 corpus → 空 stem 被自然排除
4. **实际无功能影响**：下游从不触碰这个孤立的空 TextGrid，不影响管线结果
5. **但造成不一致**：87 个 TextGrid vs 86 个其他文件，目录状态不一致

### 修改点

**AQ. `ctc_prealign.py` — TextGrid 写入移至空检测之后** (~L1193, ~L1217)

修改前：
```python
# L1193 — 空检测之前写 TextGrid
out_tg = args.output_dir / f"{stem}.TextGrid"
write_textgrid(words_pinyin, duration_s, out_tg, pauses=pauses)

# L1213 — 空检测，跳过后续文件
lab_tokens = " ".join(w["word"] for w in words_pinyin)
if not lab_tokens.strip():
    print(f"  SKIP {stem}: ASR produced no text — skipping MFA alignment")
    skipped.setdefault("empty_asr", []).append(stem)
    continue

# 写 .lab ...
```

修改后：
```python
# L1210 — 空检测，全部跳过（含 TextGrid）
lab_tokens = " ".join(w["word"] for w in words_pinyin)
if not lab_tokens.strip():
    print(f"  SKIP {stem}: ASR produced no text — skipping MFA alignment")
    skipped.setdefault("empty_asr", []).append(stem)
    continue

# L1217 — TextGrid 和 .lab 一起写入
out_tg = args.output_dir / f"{stem}.TextGrid"
write_textgrid(words_pinyin, duration_s, out_tg, pauses=pauses)

out_lab = args.output_dir / f"{stem}.lab"
out_lab.write_text(lab_tokens + "\n", encoding="utf-8")
```

### 关联样本

- `花礼_-晚间吃口小鼠_20251120_2059_365e41bc_clip0017_clip0019`：16s 纯静音/噪音，ASR 输出 `<sp1>`


## Case 25: `_inject_punctuation` 标点包含词时被误删 + `_punct_before` 漏追踪 `…` + 恢复匹配仅按时间

**日期**: 2026-07-29
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_inject_punctuation`, process_one (吞标点恢复逻辑)
**触发样本**: `花礼_呼噜呼噜小鼠毛_20250326_2058_cd0aeff6_clip0019_clip0027` (5.37s, `…` 被 `<sp2>` 替代 → mid_sp 过滤)

### 现象

CTC 预对齐在 `bu4` 和 `zhi3` 之间插入 900ms 的 `…`（省略号/长停顿标记），但最终 words tier 中 `…` 消失，被 `<sp2>` 替代，触发 `mid_sp` 过滤：

```
CTC tokens+punct:  bu4[4.35-5.37]  …[5.37-6.27]  zhi3[6.27-6.39]
MFA aligned:       ...wo3, bu4[5.49-5.60], <eps>[5.60-6.05], zhi3[6.05-6.13]...
最终 words tier:   bu4[4.35-5.37]  <sp2>[5.37-6.065]  zhi3[6.065-6.26]
                                                        ↑ … 变成 <sp2>！
```

### 根因链

**Bug A — `_inject_punctuation` 重叠判断缺分支 (~L2338-2347)**

四分支重叠判断覆盖了三种正常情况，但 `else` 分支同时包含两种相反的场景：

| 场景 | 条件 | 实际含义 | 旧行为 | 正确行为 |
|------|------|---------|--------|---------|
| punct 在 word 内 | ws≤ps ∧ we≥pe | word 包含 punct | 删除 ✓ | 删除 |
| word 左盖 punct | ws≤ps ∧ we<pe | 左侧重叠 | trim ✓ | trim |
| word 右盖 punct | ws>ps ∧ we≥pe | 右侧重叠 | trim ✓ | trim |
| ***word 在 punct 内*** | ws>ps ∧ we<pe | **punct 包含 word** | **删除 ❌** | **split** |

本例中 `…` [5.37-6.27] 包含 `zhi3` [6.065-6.26]（`5.37 < 6.065` 且 `6.27 > 6.26`），走 `else` 分支直接删除，900ms 的 `…` 全部丢失。

**Bug B — `_punct_before` 监控范围过窄 (~L4361)**

吞标点检测只追踪 `，。！？` 四种标点，`…`（省略号）被排除在外：

```python
if p["word"] not in '，。！？':   # ← …、；：等全被漏掉
    continue
```

即使 `…` 在 Phase 3.C 成功注入后被 Phase 4 吸收吞掉，也不会进入 `_swallowed_puncts`，恢复逻辑完全看不到它。

**Bug C — 吞标点恢复按时间重叠匹配而非顺序 (~L4726)**

旧代码用时间区间重叠匹配被吞标点和 `<spN>`：

```python
overlap = min(iv.xmax, p["end_s"]) - max(iv.xmin, p["start_s"])
if overlap > 0:
    iv.text = p["word"]
```

当 MFA 对齐偏差导致 `<spN>` 的边界与 CTC punct 锚点偏差较大时，重叠为 0 或负，匹配失败。

### 修改点

**AR. `_inject_punctuation` — else 分支改为 split 而非 delete** (~L2346)

修改前：
```python
            else:
                resolved[pi] = (0, 0, "", pkind)
```

修改后：
```python
            else:
                # word inside punct (ws > ps and we < pe):
                # split punct into left part (before word) + right part (after word)
                left_part  = (ps, ws, ptext, pkind)
                right_part = (we, pe, ptext, pkind)
                resolved[pi] = left_part
                if right_part[1] > right_part[0] + 0.001:
                    resolved.append(right_part)
```

效果：`…` [5.37-6.27] 被 `zhi3` [6.065-6.26] 切开 → `…` [5.37-6.065] + `zhi3` [6.065-6.26] + `…` [6.26-6.27]

**AS. `_punct_before` — 扩展追踪标点集合** (~L4361, ~L4701)

修改前：`if p["word"] not in '，。！？': continue`

修改后：`if p["word"] not in '，。！？…、；：': continue`

两处相同守卫条件同步修改。

**AT. 吞标点恢复 — 时间匹配改为按 CTC 序列顺序匹配** (~L4717-4748)

修改前按时间重叠匹配：
```python
for iv in _words_t.intervals:
    if not is_silence(iv.text): continue
    overlap = min(iv.xmax, p["end_s"]) - max(iv.xmin, p["start_s"])
    if overlap > 0: ...
```

修改后按 CTC 序列中的前后邻词匹配：
```python
# Build CTC timeline: all tokens + puncts sorted by start
_ctc_timeline = [(kind, word, start_s), ...]

for p in _swallowed_puncts:
    # Find p's neighbors in CTC sequence (prev_word, next_word)
    # Walk words tier: find <spN> between same prev_word/next_word
    for i in range(1, len(_word_ivs) - 1):
        iv = _word_ivs[i]
        if not is_silence(iv.text): continue
        if _word_ivs[i-1].text == prev_word and _word_ivs[i+1].text == next_word:
            iv.text = p['word']  # sequential match → replace
```

### 关联样本

- `花礼_呼噜呼噜小鼠毛_20250326_2058_cd0aeff6_clip0019_clip0027`：`…` [5.37s] 被 zhi3 包含 → 被删 → `<sp2>` → mid_sp 过滤

### 补充修改 (同日): 恢复逻辑中 `tokens` 变量作用域修复

修改 AT 在吞标点恢复中引用 `tokens` 变量，但该变量在 `process_one` 中名为 `ctc_tokens` 且只在 `if tokens_path.exists():` 块内定义，恢复逻辑位置在其作用域外 → 32 条文件报 `NameError: name 'tokens' is not defined`。

修复：恢复逻辑不从外部引用 `tokens`/`ctc_tokens`，改为直接从 `_tokens.jsonl` 重新读取构建 CTC timeline。

### 补充修改 (同日): BGM 检测噪声底污染 + 帧级别检测

**触发样本**: `花礼_emo来听小鼠歌吧_20250504_2058_3700c4b4_clip0003_clip0012`（`<sp1>` [0-1.71s] 内含巨响 RMS 3000-7000，但未被过滤）

**Bug D — 60 分位噪声底被高能量内容污染**:

bgm 检测用全音频 RMS 的 60 分位作为噪声底：
```python
k = max(1, int(len(all_rms) * 0.6))
nf_bgm = float(np.partition(all_rms, k)[k])
```

当 `<sp1>` 区间有语音级能量时，60 分位被拉高 → `bgm_threshold` 被拉到语音级别 → 异常静音反而检不出。**声音越大越检不出**。

修复：添加 `bgm_max_threshold`（默认 0.05）作为阈值天花板：
```python
bgm_threshold = min(bgm_threshold, args.bgm_max_threshold)
```

**Bug E — `_word_rms()` 整段平均掩盖局部高能量**:

旧代码对 silence interval 取整段 mean absolute amplitude。长静音中夹杂短时巨响时，平均值被前后静音稀释（0.111 → 0.026），不触发检测。

修复：改为 50ms 帧级别扫描，统计超阈值帧占比 ≥20% 才标记为可疑：
```python
# 50ms frames with 25ms hop
for each frame in silence interval:
    frame_energy = mean(abs( audio[frame_start:frame_end] ))
    if frame_energy > bgm_threshold:
        high_frames += 1
if high_frames / n_frames >= 0.20:
    suspect  # sustained high energy, not a transient click
```

**配置变更**:
- `run_pipeline.py` DEFAULT_CFG: 新增 `bgm_max_threshold: 0.05`
- `config.yaml`: 新增 `bgm_max_threshold: 0.05` 参考项
- `postprocess_textgrids.py`: 新增 `--bgm-max-threshold` 参数


### 补充修改 (同日): 末尾标点强制保留

**触发样本**: `直播回放_zzZ_2026年06月03日14点场_ae037890_clip0008_clip0012`（`醉…BREATHING。`→ BREATHING 吸收末尾 `。`→ 句末无标点）

**Bug F — 末尾标点被 NVV 吸收后无法恢复**:

CTC 序列 `… BREATHING 。` 中 `。` 是最后一项。Phase 4 D2 中 BREATHING 吸收尾部 `。`，吞标点恢复逻辑因 `next_word = None` 跳过末尾标点。

修复：在吞标点恢复之后、最终 QC 之前，新增**末尾标点强制保留**步骤：
1. 找 CTC 最后一个标点（按 `end_s`）
2. 若 words tier 末尾不是该标点 → 从前词末尾截取 ≥60ms
3. 截取量 = `max(0.060, CTC原始时长)` — 弹性，至少 60ms
4. 同步更新 words、hanzi、pinyin_phones 三个 tier

```python
_carve_s = max(0.060, (_last_punct_end - _last_punct["start_s"]))
# Trim last word xmax to _punct_start, append punct interval
# Sync all three tiers: words, hanzi, pinyin_phones
```

**同步修复**: 初版 hanzi tier 只 append 不 trim → 旧 BREATHING 与 `。` 重叠。改为统一处理：先 trim 末位 interval 的 xmax，再 append 标点。

### 补充修改 (2026-07-30): 标点边界保护 — 防止 dead silence 延伸覆盖标点

**触发样本**: `直播回放_三周年纪念活动进行中_2025年12月04日20点场_4d04746b_clip0010_clip0001`（`zai4` 被延伸至 5.47s 覆盖 `…` [4.95-5.43]→ `…` 被 `_inject_punctuation` 当作"word 包含 punct"删除）

**Bug G — `_refine_boundaries_by_energy` dead silence 延伸未检测与词尾重叠的标点**:

CTC 标点起点（4.95s）略早于 MFA 词尾（5.02s）→ 70ms 重叠。Dead silence 检查 `iv.xmax <= p["start_s"]` 失败，延伸未阻断 → `zai4` 延至 5.47 → Phase 3.C 中 `…` 被删。

修复：新增标点边界邻近检测：
```python
_near = abs(p["start_s"] - iv.xmax) < 0.100    # 标点起点在词尾 ±100ms 内
_body_past = p["end_s"] > iv.xmax + 0.060      # 标点主体(>60ms)在词尾之后
if _near and _body_past:
    has_punct_in_gap = True  # 阻断延伸, 标点留给 _inject_punctuation 处理
```

**参数传递链**: `_refine_boundaries_by_energy` 新增 `_punct_boundary_hits` 参数 → 由 `process_one` 传入 → 触发时记录词、标点、偏移量到 `report["punct_boundary_guard"]`。

### 补充修改 (同日): 英文 MFA 词典/G2P 路径修复

**触发样本**: 纱依100 全部 9 个英文词 `phones: []` → 英文音素全部三等分

**Bug H1 — `pretrained_models/` 在 repo 外**:

DEFAULT_CFG 中 `g2p_model: pretrained_models/g2p/english_us_arpa.zip` 相对 PROJECT_ROOT 解析 → `/mnt/project/MFA_Pause/repo/pretrained_models/...` 不存在。实际路径在 `/mnt/project/MFA_Pause/pretrained_models/...`（PROJECT_ROOT.parent）。

修复：`run_pipeline.py` 路径解析增加 fallback：
```python
if not resolved.exists() and "pretrained_models" in val:
    _parent_resolved = PROJECT_ROOT.parent / val
    if _parent_resolved.exists():
        resolved = _parent_resolved
```

**Bug H2 — G2P 失败后 `en_combined.dict` 未写入**:

`build_en_dict` L336 先定义 `combined = temp_dir / "en_combined.dict"`，但 G2P 失败时 L384 直接 `return combined` — 文件从不存在 → MFA 拿到的 dict 路径指向不存在的文件 → 全部 0 个 TextGrid → `collect_en_phones` 全部 fallback 空 phones → `_apply_en_phones` 三等分。

修复：G2P 失败或无输出时，写入纯 base dict（CMUdict）让 MFA 至少能对齐词典内已有的词：
```python
if not g2p_output.exists():
    with open(combined, 'w', encoding='utf-8') as outf:
        if base_dict.exists():
            outf.write(_read_dict(base_dict))
    return combined
```

**验证结果**: man 音素从三等分变为真实英文 MFA 对齐 `M:17% AE1:72% N:12%`。

### 补充修改 (同日): `_NVV_PATTERN` 重复包裹 `<>`

**触发样本**: `直播回放_三周年纪念活动进行中_2025年12月07日19点场_7bd7ea1c_clip0007_clip0013`（pinyin_phones tier 中 `<CONFIRMATION-EN>` 变成 `<<CONFIRMATION-EN>>`）

**Bug I — `_NVV_PATTERN` 匹配已包裹的 NVV token**:

`_NVV_PATTERN` 的 lookbehind `(?<![A-Za-z-])` 和 lookahead `(?![A-Za-z-])` 不排斥 `<>`。`_finalise_textgrid` L142 用 `.sub()` 包裹未包裹的 NVV 时，已包裹的 `<CONFIRMATION-EN>` 被再次匹配 → `<<CONFIRMATION-EN>>`。

words/hanzi tier 不受影响（只在 `_finalise_textgrid` 中包裹一次），但 pinyin_phones tier 在 `build_pinyin_phones_tier` 和后续 `_sync_derived_tiers` 之间被多次调用 `.sub()`，造成累积。

修复：lookbehind 和 lookahead 均加入 `<>` 排斥：
```python
# 旧: (?<![A-Za-z-]) ... (?![A-Za-z-])
# 新: (?<![A-Za-z<>-]) ... (?![A-Za-z<>-])
```

已包裹的 token 不再被匹配，未包裹的仍正常包裹。`build_pinyin_phones_tier` 的 `strip('<>')` 负责修复已有双重包裹数据。


## Case 26: pinyin_phones 声母独占总时长 → 韵母消失 (ch/zh/sh → 缺 final)

**日期**: 2026-08-04
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `build_pinyin_phones_tier`
**触发样本**: shayi_huali/纱依/output/ ~15.5% 的 zh/ch/sh 声母词受影响

### 现象

words tier 中拼音词为 `chang4`，但 pinyin_phones tier 只有声母 `ch` 占据整词时长，
韵母 `ang4` 完全消失：

```
words tier:       chang4  [8.850s - 9.210s]  dur=360ms
pinyin_phones:    ch      [8.850s - 9.210s]  dur=360ms  ← ang4 丢失!
```

同样问题影响所有 zh/ch/sh 声母词：`zhi3`→只有 `zh`、`shi4`→只有 `sh`、`zhong4`→只有 `zh`。

还存在另一种表现 FULL_WORD_AS_PHONE：当 MFA 完全没产出任何音素时，
整词作为单个 phone 写入（如 `chang4 [全区间]`），也失去了声母/韵母拆分。

### 范围

对 shayi_huali/纱依/output/ 2894 个文件的采样扫描（每 3 取 1，共 965 文件）：

| 类型 | 数量 | 占比 |
|------|------|------|
| MISSING_FINAL（声母独占，韵母消失） | 601 | 8.7% |
| FULL_WORD_AS_PHONE（整词作为单音素） | 454 | 6.5% |
| 正常拆分 | 5861 | 84.5% |

按声母: `sh` 538 > `zh` 380 > `ch` 156。

### 根因链

两种子类型共享同一个根因 — MFA 对该音节的音素产出不足，而代码在音素不足时
只取字典第一个条目（声母）覆盖全区间：

**子类型 A — MISSING_FINAL** (原 line 562-564):

1. MFA 对齐 `chang4` 时只产出 **1 个** IPA 音素（声母 IPA），未将韵母区分为独立音素
2. 代码查字典 `fullpinyin_enword.dict` → `dict_phones = ['ch', 'ang4']`（2 个）
3. `word_phones` 只有 1 个 MFA 音素 → 条件 `len(word_phones) <= 1` 为 True
4. 旧代码: `dict_phones[0]` (`'ch'`) 分配给整词区间 → `'ang4'` 彻底丢失

**子类型 B — FULL_WORD_AS_PHONE** (原 line 494-496):

1. MFA 对该词完全没产出音素（`word_phones` 为空），或被泄漏音素过滤清空
2. 旧代码: 直接用词文本 `'chang4'` 作为单音素兜底，未查字典拆分
3. 损失了声母/韵母的时间分辨率（对于 TTS 训练，单音素 `chang4` 不如 `ch` + `ang4`）

### 修改点

**BA. `build_pinyin_phones_tier` — 字典查前移 + 空音素时按比例拆分 (~line 494-527)**

字典查询从原 line 498 移至 `word_phones` 空检查之前，使空音素场景也能利用字典信息。
当 `dict_phones >= 2` 且 `word_phones` 为空时，按 35:65 比例拆分词区间为声母+韵母，
每段地板 30ms。NVV/English/punct 不进入此分支（保留旧兜底行为）。

**BB. `build_pinyin_phones_tier` — 单音素场景按比例拆分 (~line 584-616)**

原代码 `len(dict_phones) == 1 or len(word_phones) <= 1` 将"零声母"和"音素不足"
混为一谈。修复后拆分为三个分支：
- `len(dict_phones) == 1`: 零声母 → 单音素覆盖全区间（保留旧行为）
- `len(word_phones) >= 2`: MFA 足量音素 → 用 MFA 边界精确拆分（保留旧行为）
- 否则（`dict_phones >= 2` 但 `word_phones <= 1`）: MFA 音素不足 → 按比例拆分

### 修改前

```python
# (原 line 494) 空音素 → 整词兜底，不查字典
if not word_phones:
    new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, word))
    continue

# ... 字典查询在下方的 line 498 ...

# (原 line 562-564) 零声母和音素不足混为一谈
if len(dict_phones) == 1 or len(word_phones) <= 1:
    new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, dict_phones[0]))
```

### 修改后

```python
# 字典查询已移至 word_phones 空检查之前

# 空音素 → 字典有 2+ 条目时按比例拆分
if not word_phones:
    if dict_phones and len(dict_phones) >= 2 and not punct/NVV/English:
        word_dur = w_iv.xmax - w_iv.xmin
        _init_frac = 0.35; _min_seg = 0.030
        split = w_iv.xmin + max(_min_seg, word_dur * _init_frac)
        split = min(split, w_iv.xmax - _min_seg)
        new_intervals.append(Interval(w_iv.xmin, split, dict_phones[0]))
        new_intervals.append(Interval(split, w_iv.xmax, dict_phones[1]))
    else:
        new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, word))
    continue

# 三分支: 零声母 | MFA足量 | MFA不足→比例拆分
if len(dict_phones) == 1:
    new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, dict_phones[0]))
elif len(word_phones) >= 2:
    # MFA 边界精确拆分（不变）
    ... 
else:
    # dict_phones >= 2 但 word_phones <= 1 → 比例拆分（同上）
    ...
```

### 比例选择依据

- 35:65 是汉语声母:韵母的典型时长比（塞擦音 ch/zh ~30-40%，擦音 sh ~25-35%）
- 30ms 地板防止极短音节出现零时长段
- 音节 < 60ms 时退化为 50:50 均分

### 关联样本

- `直播回放_3D初披露--_2024年2月29日20点场_97187255_clip0012_clip0000.TextGrid`:
  `chang4` [8.850-9.210] pinyin_phones 只有 `ch`，缺 `ang4`
- `直播回放_zzZ_2026年06月07日20点场_1341f0ea_clip0012_clip0013.TextGrid`:
  4 个 `chang4` 中 2 个异常（1 个 MISSING_FINAL + 1 个 FULL_WORD_AS_PHONE）

### 补充修改 (2026-08-04): 按声母类型细化拆分比例

原修复使用统一 35% 拆分声母:韵母。对塞音 (b/p/d/t/g/k, ~15-25%) 和鼻音/边音
(m/n/l, ~15-25%) 偏高。新增模块级 `_INIT_FRAC` 字典 (~line 434)，按声母类型返回
对应比例：塞音 0.20、鼻音/边音 0.22、擦音 0.28、塞擦音 0.35 (默认)。

### 关联修改点

| 修改 | 位置 | 作用 |
|------|------|------|
| BA  | `build_pinyin_phones_tier` ~line 510-527 | FULL_WORD_AS_PHONE: 字典前移 + 比例拆分 |
| BB  | `build_pinyin_phones_tier` ~line 600-616 | MISSING_FINAL: 三分支替代 OR 条件 |
| BC  | 模块级 `_INIT_FRAC` ~line 434 | 按声母类型细化拆分比例 |
| BD  | `process_one` ~line 5350 前 (新增) | 质检过滤: `init_only_phone` 残留检测 |

---

## Case 27: words tier 区间重叠 — 相邻词时间边界交叉

**日期**: 2026-08-04
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `_fix_overlapping_boundaries` (新增), `process_one` (过滤)
**触发样本**: shayi_huali/纱依/output/ ~9.7% 文件存在不同程度的重叠

### 现象

words tier 中相邻 interval 的 xmax > 下一 interval 的 xmin，即时间区间重叠：

```
轻度 (< 20ms):
  ye3 [7.349-7.554] ↔ yi3 [7.537-7.657]  overlap=17ms

中度 (标点侵入词):
  ，[3.990-4.740] ↔ zuo2 [4.600-4.740]    overlap=140ms

重度 (双重标点):
  ，[4.850-9.360] ↔ ，[5.100-9.360]       overlap=4260ms
  ，[5.100-9.360] ↔ y   [5.455-5.610]     overlap=3905ms
```

### 根因链

**子类型 A1 — 轻度边界重叠 (< 30ms)**:
1. MFA 帧精度 (10ms) + `_snap_to_ctc` 将不同词 snap 到不同 CTC 锚点
2. 相邻词的 MFA 边界和 CTC 锚点来自独立的决策，未做连续性校验
3. 结果：prev.xmax (来自 MFA/CTC 决策 A) > next.xmin (来自决策 B)

**子类型 A2 — 标点重叠**:
1. `_inject_punctuation` 注入标点时使用了过宽的区间范围
2. 连续标点未做去重或裁剪，导致多个标点 interval 使用相同或高度重叠的范围
3. 标点 xmax 延伸到后续内容词内部

**子类型 A3 — 极短词挤压**（与 Case 29 同源）:
1. 极短词 (< 30ms) 挤在两个正常词之间
2. 因为太短，其 xmin < prev.xmax（物理上必然重叠）

### 修改点

**CA. `_fix_overlapping_boundaries` — 新增函数，修复轻度边界重叠** (~line 新增)

对 words tier 扫描相邻 interval 重叠：
- 两内容词重叠 < 30ms → 取中点分界 (`split_overlap`)
- 内容词与标点重叠 → 标点裁短 (`clip_punct`)
- 重叠 ≥ 30ms 或涉及静音/多重重叠 → 不修复，交给过滤

修复后调用 `_sync_derived_tiers` 同步 hanzi + pinyin_phones。

**CB. `process_one` — 过滤: `overlapping_words`** (~line 5350 前)

修复后仍存在重叠的 → 添加 `overlapping_words` 过滤原因，文件进 `filtered/`。

### 关联样本

- `0-_杂谈_月夜下的魔法絮语_5058f1af_clip0002_clip0005.TextGrid`:
  3 处重叠 (17ms/103ms/7ms)
- `0-_杂谈_月夜下的魔法絮语_5058f1af_clip0002_clip0012.TextGrid`:
  3 处重度重叠 (140ms/424ms/596ms)

---

## Case 28: 倒置 interval — xmin > xmax

**日期**: 2026-08-04
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (过滤，不修复根因)
**触发样本**: shayi_huali/纱依/output/ 个别文件

### 现象

三个 tier 同时出现 xmin > xmax 的倒置 interval：

```
hanzi:          '这'  [1.0283-1.0200]  dur=-8.3ms
words:          'zhe4' [1.0283-1.0200]  dur=-8.3ms
pinyin_phones:  'zhe4' [1.0283-1.0200]  dur=-8.3ms
```

相邻词正常：`ge4 [1.0200-1.1100] dur=90ms`。

### 根因链

1. `_refine_boundaries_by_energy` 或 `_inject_punctuation` 将 `zhe4.xmin` 推到 1.0283
2. `zhe4.xmax` 被锚定为 `ge4.xmin = 1.0200`（下一词的起点）
3. 词首前推 + 词尾被后邻词锚定 → xmin > xmax 倒置
4. Case 12 的修复 AC 只覆盖 `_snap_to_ctc` 重叠路径的倒置，此处的倒置走另一条路径
5. 倒置 interval 通过 `_sync_derived_tiers` 同步到三个 tier

### 修改点

**DA. `process_one` — 过滤: `inverted_interval`** (~line 5350 前)

扫描所有 tier 的所有 interval，检测 `xmin > xmax + 0.001`。
存在倒置 → 添加 `inverted_interval` 过滤原因。

不修复根因：涉及能量调整和标点注入的深层交互，风险较高。
修复线索：在 `_refine_boundaries_by_energy` 词首前拉逻辑中添加 xmin ≤ xmax - 40ms 约束。

### 关联样本

- `直播回放_三周年纪念直播_2025年11月25日21点场_c2ad2c62_clip0010_clip0018.TextGrid`:
  zhe4/这 xmin=1.0283 > xmax=1.0200

---

## Case 29: 极短内容词 — (< 30ms) 物理不可能

**日期**: 2026-08-04
**涉及文件**: `scripts/postprocess_textgrids.py`
**涉及函数**: `process_one` (过滤，不修复根因)
**触发样本**: shayi_huali/纱依/output/ ~91/414 文件存在极短词

### 现象

words tier 中存在 < 30ms 的内容词（非标点、非静音）：

```
ke3  [7.554-7.564]  dur=10ms   ← 物理上不可能
shi2 [10.174-10.184] dur=10ms
ting2 [4.979-5.017]  dur=38ms  ← 边界值
doing [4.480-4.510]  dur=30ms  ← 英文词
```

30ms 以下的音节在生理上不可能 — 汉语单音节最短约 60-80ms，英语最短约 80-100ms。

### 根因链

1. `_snap_to_ctc` 将相邻词边界向 CTC 锚点靠拢
2. 中间词被挤压到剩余空间 → 可能被压缩到 < 30ms
3. 极短词引发 Case 27 的子类型 A3（重叠），产生连锁问题
4. 英文词 `doing` 30ms 可能是 ASR 误检 + MFA 无法对齐

### 修改点

**EA. `process_one` — 过滤: `short_word`** (~line 5350 前)

扫描 words tier，检测内容词（非 punct/NVV/silence）时长 < 30ms。
存在极短词 → 添加 `short_word` 过滤原因。

不修复根因：涉及 `_snap_to_ctc` 边界压缩逻辑和 CTC 锚点质量，
简单合并到相邻词可能引入新问题。建议作为后续优化通过改进边界调整逻辑来根治。

### 关联样本

- `0-_杂谈_月夜下的魔法絮语_5058f1af_clip0002_clip0005.TextGrid`:
  ke3 10ms, shi2 10ms
- `0-_杂谈_月夜下的魔法絮语_5058f1af_clip0005_clip0021.TextGrid`:
  doing 30ms

## Case 30: 参考文本模糊子串匹配 — 纯英文/混杂英文 CTC 锚点标定风险分析

**日期**: 2026-08-04
**涉及文件**: `scripts/postprocess_textgrids.py`, `scripts/adjust_ctc_boundaries.py`, `scripts/pipeline_utils.py`
**涉及函数**: `_word_matches`, `_alpha_text_matches`, `_align_word_sequences`, `_snap_to_ctc`, `_normalize_word_spellings`, `_build_hanzi_tier`, `align_sequences`
**触发条件**: 参考文本含英文单词（纯英文或中英混杂），且英文词长度 ≤3 字符或为其他词的子串

### 现象

Post-MFA 参考文本匹配链路中，存在两套不同的对齐策略用于英文 token：

| 阶段 | 函数 | 匹配策略 | 使用位置 |
|------|------|---------|---------|
| CTC 锚点边界对齐 | `align_sequences` (pipeline_utils.py:963) | **精确匹配** (`a==b`) | `_snap_to_ctc` (postprocess_textgrids.py:3460) |
| 拼写规范化 + hanzi 构建 | `_word_matches` / `_alpha_text_matches` (postprocess_textgrids.py:1347/1450) | **模糊子串** (`c in r or r in c`) | `_normalize_word_spellings` (postprocess_textgrids.py:1657), `_build_hanzi_tier` (postprocess_textgrids.py:1571) |

这两套策略在英文 token 上的行为不一致：

```
# align_sequences (精确匹配):
CTC "Pop" vs MFA "Pop" → match ✓
CTC "Pop" vs MFA "K-Pop" → NO match ✗  (子串但不等)
CTC "Up"  vs MFA "Up"  → match ✓

# _word_matches (模糊子串):
CTC "Pop" vs ref "K-Pop" → match ✓  (子串包含)
CTC "Up"  vs ref "V-Up"  → match ✓  (子串包含)
CTC "a"   vs ref "cat"   → match ✓  (单字母=任意包含)
CTC "he"  vs ref "the"   → match ✓  (子串包含)
```

### 根因链

**A. 精确匹配层 (`_snap_to_ctc`) 的问题:**

1. `align_sequences`(pipeline_utils.py:981) 仅接受 `cost=0 if a[i-1]==b[j-1] else 1`
2. 英文词在 CTC 输出中常以原词出现（如 `Pop`, `Up`），但若 MFA 将其视为 OOV 输出 `<unk>`，精确匹配失败
3. 若 CTC token 计数 ≠ MFA token 计数 → 进入 NW 对齐（postprocess_textgrids.py:3456-3460）→ 精确匹配找不到对应 → 英文 token 无 CTC 锚点 → 回退到 MFA 边界（postprocess_textgrids.py:3500）
4. **英文 token 无条件信任 CTC**（postprocess_textgrids.py:3534-3535: `use_mfa=False`），但若 CTC token 已因精确匹配失败被标记为 `None`，则 MFA 边界被保留——产生矛盾

**B. 模糊子串层 (`_word_matches` / `_alpha_text_matches`) 的问题:**

1. **核心**: 子串包含 `c in r or r in c`（postprocess_textgrids.py:1370）过于宽松
   - `"Pop"` in `"K-Pop"`? `"Pop"` 不在 `"k-pop"` 中！——不匹配 ✗
   - `"Pop"` in `"Pop"`? 自身包含 ✓
   - 实际上 `"Pop" in "K-Pop"` → False (因为 `-Pop` 不包含不含连字符的 Pop)... 

   等等，`"pop" in "k-pop"` → True! 因为 lowercase 后 `"k-pop"` 包含 `"pop"`。所以 `-` 不干扰子串匹配。
   
   真正的风险在于短词:
   - `"a"` 匹配任何含 `a` 的英文词（`"cat"`, `"day"`, `"and"` 等）
   - `"he"` 匹配 `"the"`, `"there"`, `"where"` 等
   - `"or"` 匹配 `"for"`, `"more"`, `"word"` 等
   - `"in"` 匹配 `"win"`, `"within"`, `"inside"` 等

2. **单字母 CTC token 显式允许**（postprocess_textgrids.py:1374-1375）:
   ```python
   if len(c) == 1 and c.isalpha():
       return c in r     # 单字母匹配任意含该字母的参考词
   ```

3. **NW gap-first 回溯**（postprocess_textgrids.py:1426-1431）: 当多个模糊匹配代价相同时（均为 0），优先丢弃 CTC token（gap），保留参考词。这在英文碎片化场景中是正确的（CTC token 更多），但在短词歧义场景中可能丢弃正确的 CTC token。

4. **贪婪 alpha 消费**（postprocess_textgrids.py:1576-1582）: `continue_match` 使用相同子串逻辑，可能过度消费参考 alpha 单元。

**C. CTC 锚点标定 (`adjust_ctc_boundaries`) 层面：无参考文本参与**

`adjust_ctc_boundaries.py` 的 `adjust_boundaries()` (adjust_ctc_boundaries.py:121) 纯能量驱动:
- 英文词通过 `_is_nvv` 过滤（adjust_ctc_boundaries.py:127）— 仅跳过 NVV token
- 普通英文词仍会被 `_search_energy_rise` / `_search_energy_fall` 修正边界
- **不涉及参考文本，不存在文本混乱风险** ✅

### 针对当前数据集 (hecheng_english_mfa) 的实际风险评估

数据集特征（`/mnt/local_E/Voxcpm/output/generated_scripts.jsonl`）:
- 54000 条中英混杂文本
- 英文词: 仅 1021 条含英文，绝大多数 1 个英文词/条
- 英文词长: 仅 2-3 字符（Pop, Up, PV, MV, AI, SOS, BGM, GPT, ...）
- 无 ≥5 字母的英文词，无 3+ 英文词的条目
- 英文词以品牌名/缩写为主，非自然语言句子

**针对此数据集的结论**:
1. 英文词均为短缩写/品牌名 → 子串匹配风险较低（词形独特，不太可能嵌套）
2. 但 `"Up"` 匹配 `"V-Up"` 中的 `"up"` 子串 ✓ 正确，但也匹配 `"Upload"`, `"Popup"` 等 —— 当前数据集不含这些词
3. 单字母 token 不会出现（CTC 模型不会把 "K-Pop" 拆成单字母）
4. **主要风险**: CTC 中文模型对英文词的边界标定本身不准（duration 偏长/偏短），而非文本匹配问题

**普遍性纯英文文本的风险**（若后续扩展数据集）:
1. 短词子串歧义: high risk
2. NW gap-first 可能丢弃正确 CTC token: medium risk
3. `_snap_to_ctc` 精确匹配 + 无条件 CTC 的矛盾: needs attention
4. `adjust_ctc_boundaries` 能量修正无风险: safe ✅

### 修改点

暂无代码修改。此 Case 为风险分析记录，供后续纯英文文本扩展时参考。

**潜在改进方向**:
- `_word_matches`: 对纯英文参考文本，可将 `c in r or r in c` 改为优先精确匹配，仅在不匹配时 fallback 到子串
- `_snap_to_ctc`: 当 `ctc_aligned[idx] is None` 且 `is_english_token(mfa_iv.text)` 时，应警告而非静默使用 MFA 边界
- 可增加 QC 规则: 检测英文词的 CTC 时长是否在合理范围 (80ms-2000ms)

### 验证方法

```python
# 测试 1: 检查英文词的 CTC-MFA 边界偏差
for tok in ctc_tokens:
    if is_english_token(tok["word"]):
        dur = tok["end_s"] - tok["start_s"]
        assert 0.08 <= dur <= 2.0, \
            f"English word {tok['word']} CTC duration {dur:.3f}s out of range"

# 测试 2: 检查 _word_matches 子串假阳性
# 构造 (ctc_token, ref_unit) 对照表，检查匹配结果
test_cases = [
    ("he", "the", True, "substring — 可接受的假阳性"),
    ("a", "cat", True, "单字母 — 当 CTC 碎片化时正确，否则假阳性"),
    ("or", "for", True, "substring — 假阳性"),
    ("in", "win", True, "substring — 假阳性"),
]
for ctc, ref, expected, note in test_cases:
    actual = _word_matches(ctc, ref)
    if actual != expected:
        print(f"MISMATCH: _word_matches({ctc!r}, {ref!r}) = {actual} (expected {expected}) — {note}")
```

### 关联样本

- 数据集: `/mnt/Raw/新版合成英文数据` (54000 条, 1021 条含英文)
- JSONL: `/mnt/local_E/Voxcpm/output/generated_scripts.jsonl`

### 实测结果 (2026-08-04, 10 条样本 CTC prealign)

**测试命令**:
```bash
python3 scripts/prepare_english_tts.py \
  --jsonl /mnt/local_E/Voxcpm/output/generated_scripts.jsonl \
  --audio-root /tmp/test_en_ctc/audio_data \
  --output-dir /tmp/test_en_ctc/ctc_pretg --limit 10

python ctc_prealign.py \
  --data-dir /tmp/test_en_ctc/audio_data \
  --pinyin-dir /tmp/test_en_ctc/ctc_pretg \
  --output-dir /tmp/test_en_ctc/ctc_pretg \
  --model-path .../Multilingual-NVASR --device cuda:0 --limit 10
```

**10 条样本: 全部通过** — 无词序错位、无 token 重叠、无时间戳倒序。

**发现的真实问题（非参考文本混淆，而是 CTC 分词器行为）**:

| 问题 | 样本 | 详情 |
|------|------|------|
| 英文词碎片化 | 000021 | `"SOS"` → CTC 输出 `"SOS"` (180ms) + `"OS"` (420ms)，normalize_en 未合并残留 `"OS"` |
| 英文词碎片化 | 000135 | `"fan"` → CTC 输出 `"f"` (60ms) + `"an"` (60ms)，极短碎片 |
| 英文词碎片化 | 000248 | `"show"` → CTC 输出 `"s"` (60ms) + `"how"` (360ms) |
| 英文词碎片化 | 000352 | `"fan"` → CTC 输出 `"f"` (60ms) + `"an"` (300ms) |
| normalize_en 过度合并 | 000044 | `"In"+"s"+"ta"+"gram"` → `"Instagramsta"` (多出 `"sta"`) |
| normalize_en 误合并 | 000021 | `"li"+"ve"` → `"leave"` (应为 `"live"`) |
| 拼音误判为 NVV | 000021 | `"zan2"` → `"BREATHING"` (240ms) |
| 拼音误判为 NVV | 000248 | `"jin1"` → `"BREATHING"` (180ms) |

**结论**: 对于此数据集（中英混杂，英文为 2-3 字母品牌名），CTC 锚点标定的参考文本匹配**没有导致错位**。实际风险在于：
1. **CTC 分词器碎片化** — 中文 CTC 模型将英文词拆成碎片，normalize_en 无法全部修复
2. **拼音→NVV 误判** — 短拼音音节被 NVASR 误识别为 NVV token
3. **纯英文场景的子串匹配风险**在本数据集中未触发（英文词太少太短），但仍需关注

---

## Case 31: 英文 CTC 锚点三重修复 — 碎片化 / 合并错误 / NVV 误判

**日期**: 2026-08-04
**涉及文件**: `scripts/normalize_english_tokens.py`, `scripts/ctc_prealign.py`
**涉及函数**: `_token_matches_ref`, `normalize_stem`, `make_patched_inference`
**触发样本**: 实测 10 条样本中发现 3 类共性问题

### 问题全景

```
                    ┌──────────────────────────┐
                    │ NVASR ASR (SenseVoice)   │
                    │ tokenizer: 中文词表为主    │
                    │ OOV English → 碎片化       │
                    └──────────┬───────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
      ┌──────────┐     ┌────────────┐     ┌──────────────┐
      │ 碎片化    │     │ ASR 幻觉    │     │ NVV 误判     │
      │ fan→f+an │     │ Instagram  │     │ zan2→BREATH  │
      │ SOS→SOS+ │     │ →Instagra │     │ jin1→BREATH  │
      │    OS    │     │   msta     │     │              │
      └────┬─────┘     └─────┬──────┘     └──────┬───────┘
           │                 │                   │
           ▼                 ▼                   ▼
    normalize_en       normalize_en         CTC logits
    合并碎片          用错误参考合并        blank bias=4.0
    f+an→fan          In+s+ta+gram        → NVV token
    SOS+OS→???        →Instagramsta
```

### 根因分析 (逐问题)

---

### 问题 1: 英文词碎片化 — CTC tokenizer OOV 拆字

**现象**:

| 样本 | 参考英文词 | CTC 碎片 | normalize_en 结果 | 状态 |
|------|-----------|---------|-------------------|------|
| 000021 | SOS | `SOS`(180ms) + `OS`(420ms) | 残留 `OS` | ❌ 未修复 |
| 000135 | fan | `f`(60ms) + `an`(60ms) | 残留 `f`+`an` | ❌ 未修复 |
| 000248 | show | `s`(60ms) + `how`(360ms) | 残留 `s`+`how` | ❌ 未修复 |
| 000352 | fan | `f`(60ms) + `an`(300ms) | 残留 `f`+`an` | ❌ 未修复 |
| 000021 | live | `li`(180ms) + `ve`(→leave) | `leave` | ❌ 错误合并 |
| 000125 | live | `li` + `ve` | `live` | ✅ 正确 |
| 000223 | deadline | `de` + `ad` + `line` | `deadline` | ✅ 正确 |
| 000223 | live | `li` + `ve` | `live` | ✅ 正确 |

**根因链**:

1. SenseVoice tokenizer 词表以中文为主，OOV 英文词无对应 token
2. CTC 解码时将英文词拆成 BPE 级子词碎片（`fan` → `f` + `an`、`show` → `s` + `how`）
3. `normalize_english_tokens.py` 通过 NW 对齐 + 参考文本匹配合并碎片
4. 但合并依赖两个条件同时满足：
   - **条件 A**: `_token_matches_ref` 返回 True（子串包含或拼音→英文）
   - **条件 B**: `all_fragments` 检查通过（每个碎片至少与目标词共享一个字母）
5. 条件 A 的拼音→英文匹配（normalize_english_tokens.py:126）无条件 `return True`，过于宽松
6. 当碎片是英文子串（如 `f` 在 `fan` 中）时条件 A 通过，但 `an` 也是独立英文词（`is_english_token("an")=True`），`an in "fan"` → True
7. **关键缺陷**: `all_fragments` 检查（normalize_english_tokens.py:264-265）将已经是完整英文词的 fragment 当作"substring of target"放行:
   ```python
   elif is_english_token(t) and t.lower() in en_lower:
       pass  # substring of target (e.g. "play" in "cosplay")
   ```
   但 `"OS"` 是英文词且 `"os" in "sos"` → True → 被当作 target 的子串放行
   然而 `"SOS"` 才是参考词，`"OS"` 是残留碎片——这个逻辑仅检查"fragment 是否是 target 的子串"，不检查"reference word 是否完整"

8. 对于 `fan` → `f` + `an`：`"f" in "fan"` ✓, `"an" in "fan"` ✓ → 理论上能合并为 `fan`，但 normalize_en 没有触发。排查：`extract_word_chars("今天天气超好，心情也超好，作为一个fan老用户...")` → `["今","天","天","气","超","好"，"心","情","也","超","好"，"作","为","一","个","fan","老","用","户",...]` → `fan` 被识别为英文参考词 ✓。但 `_token_matches_ref("an", "fan")` 返回 True（`"an" in "fan"`），`all_fragments` 检查 `"f" in "fan"` ✓，`is_english_token("an") and "an" in "fan"` ✓ → 应该触发合并。

   **实际未触发的原因**: NW 对齐时，`fan` 参考词与 `.lab` tokens 的 DP 匹配可能将 `f` 和 `an` 分别对齐到了不同的参考词位置。`.lab` 中 `fan` 位置前后是 `ge4 f an lao3`，而参考词序列是 `[..., "fan", ...]`。`f` 可能和前面的 pinyin 对齐了（pinyin→English 无条件 `return True`），导致 `f` 没有被分配到 `fan` 的 ref 索引下。

**设计修复 — Fix 1: `normalize_english_tokens.py` 增加碎片回收 Pass**:

在 `normalize_stem` 末尾增加 **Pass 2: 碎片回收** — 对 normalize_en 后仍然残留的英文碎片进行二次回收:

```python
# Pass 2: Fragment reclamation (NEW)
# After Pass 1 merge, check for orphan fragments: single-letter ASCII
# tokens or short English tokens (< 3 chars) adjacent to merged English
# words.  Absorb them into the nearest English token.
def _reclaim_fragments(lab_tokens, ctc_tokens):
    """Merge orphan English fragments into adjacent English tokens."""
    # Identify English tokens and their fragments
    # Rule: if a fragment is a substring of an adjacent English word,
    # merge it into that word's time range.
    ...
```

**实现要点**:
1. 扫描 `.lab` tokens，找到所有 `is_english_token(t)` 的 token
2. 对于每个长度 ≤2 的英文 token，检查其前后邻接是否有长度 ≥3 的英文 token
3. 如果短 token 是长 token 的子串 → 合并
4. 如果两个相邻的短 token 可以拼成完整英文词 → 合并
5. 同步更新 `_tokens.jsonl` 的时间戳（start=min, end=max）

**涉及文件**: `scripts/normalize_english_tokens.py`
**涉及函数**: `normalize_stem` (新增 `_reclaim_fragments` 调用)

---

### 问题 2: normalize_en 合并错误 — 参考文本污染 + 过度宽松匹配

**现象**:

| 样本 | ASR 原始输出 (_text_raw.txt) | 参考词 | normalize_en 结果 |
|------|---------------------------|--------|-------------------|
| 000044 | `搭配 Instagramsta效果满分` | `Instagramsta` (错误!) | `In`+`s`+`ta`+`gram` → `Instagramsta` |
| 000021 | `leave走起` | `leave` (应为 `live`) | `li`+`ve` → `leave` |

**000044 "Instagramsta" 根因链**:

1. NVASR ASR 模型听 "Instagram" → 输出 "Instagramsta"（幻觉，多出 "sta"）
2. `_text_cn.txt` 被写为 ASR 输出文本（含 "Instagramsta"）
3. CTC forced alignment 将 "Instagramsta" 拆成 BPE 碎片: `In` + `s` + `ta` + `gram`
4. `normalize_stem` 读 `_text_cn.txt` → `extract_word_chars` → 参考英文词 = `Instagramsta`
5. NW 对齐: 碎片 ↔ `Instagramsta` → 合并 → `.lab` 写入 `Instagramsta`
6. **最终结果**: 参考文本被 ASR 幻觉污染，经 normalize_en 固化到 `.lab` 和 tokens

**根本原因**: `_text_cn.txt` 的内容来自 ASR 输出而非原始 JSONL 参考文本。`prepare_english_tts.py` 写入的原始 `_text_cn.txt` 是正确的（来自 JSONL），但 `ctc_prealign.py` 的 `make_patched_inference` 将 ASR 文本写入了 `_text_cn.txt`，覆盖了原始文本。

**000021 "leave" vs "live" 根因链**:

1. ASR 输出 "leave" (误听)
2. CTC 碎片: `li` + `ve`
3. 参考词: `leave` → normalize_en 合并 = `leave`
4. 实际应该是 `live`

这两个案例的共同根因: **参考文本来自 ASR 输出而非原始标注**。

**设计修复 — Fix 2: 参考文本源头保护 + 匹配约束**:

**2a. `ctc_prealign.py` 保护原始 `_text_cn.txt`**:

`make_patched_inference` (ctc_prealign.py:279-281) 在有关联文本时使用 `ref_texts[stem]`:
```python
if stem in ref_texts:
    align_text = ref_texts[stem].strip()
```

但 `ref_texts` 来自 `_text_cn.txt` 文件，而这些文件在 pipeline 启动时可能已被之前的运行污染。

**修改**: 在 `make_patched_inference` 中，当 ref_texts 可用时，ASR 解码结果应**仅用于 NVV/标点检测**，英文词应**保留 ref_texts 中的原始英文词**:

```python
# NEW: Cross-check ASR English tokens against reference text
# If ref_text has an English word at the same position, use ref version
ref_eng_words = re.findall(r'[a-zA-Z]{2,}', ref_text)
asr_eng_words = re.findall(r'[a-zA-Z]{2,}', asr_text)
if set(asr_eng_words) != set(ref_eng_words):
    # ASR hallucinated English → revert to reference
    ...
```

**2b. `normalize_english_tokens.py` 拼音→英文匹配增加元音约束**:

`_token_matches_ref` (normalize_english_tokens.py:126-127):
```python
# Before (unconditional):
if len(t) >= 2 and t[-1].isdigit() and t[:-1].isalpha():
    return True

# After (with vowel guard, same as Case 10 fix):
if len(t) >= 2 and t[-1].isdigit() and t[:-1].isalpha():
    vowel_count = sum(1 for ch in r if ch in 'aeiou')
    return vowel_count >= 2
```

这防止短英文词（`OH`, `OP`, `Up`, `in` 等 ≤1 元音）被 pinyin 音节错误匹配。

**2c. `normalize_english_tokens.py` 增加合并结果验证**:

在 `normalize_stem` 末尾增加验证:
```python
# NEW: Verify merged result matches reference
for en_word, indices in changes:
    current = [lab_tokens[i] for i in indices]
    merged = "".join(current)
    # If merged letters don't form the reference word, warn
    if sorted(merged.lower()) != sorted(en_word.lower()):
        print(f"  [WARN] {stem}: merged '{merged}' vs ref '{en_word}' — possible hallucination")
```

**涉及文件**: `scripts/normalize_english_tokens.py` (2b, 2c), `scripts/ctc_prealign.py` (2a)

---

### 问题 3: 拼音→NVV 误判 — blank-frame bias 过度激进

**现象**:

| 样本 | CTC 片段 | NVV 分类 | 实际应为 | 时长 |
|------|---------|---------|---------|------|
| 000021 | `zan2` | `BREATHING` | 拼音 `zan2` (咱) | 240ms |
| 000248 | `jin1` | `BREATHING` | 拼音 `jin1` (今) | 180ms |

**根因链**:

1. SenseVoice 模型有 30 类 NVV token (Breathing, Laughter, Crying, ...) 位于 token ID 25025-25054
2. `make_patched_inference` (ctc_prealign.py:218-221) 对 CTC blank 帧施加 NVV bias:
   ```python
   top_pred = x.argmax(dim=-1)
   is_blank = (top_pred == BLANK_ID)
   x[is_blank, NVV_START:NVV_END + 1] += bias_value  # default 4.0
   ```
3. NVV_BIAS_DEFAULT = 4.0 → logit 加 4.0 → softmax 后概率 ≈ 0.98
4. 当短拼音音节（如 `zan2` 240ms）前后有 silence/blank 帧时：
   - 拼音音节本身被正确解码
   - **但**周围的 blank 帧被 biased → NVV token 概率飙升
   - CTC greedy decoding 选择了 NVV token 而非 blank → 在音节边界插入 NVV
5. 更关键的是: NVV bias 作用于 **所有 blank 帧**，不区分"真正的 silence"（数百ms静音）和"音节间的短暂 blank"（60-120ms 间隔）
6. `zan2` [9.27-9.51] 后的空白帧在 `pause_threshold` (8帧≈480ms) 内被检测为 NVV

**为什么 `zan2` → BREATHING 而不是其他 NVV**:
- CTC blank 帧被 bias 后，NVVS_START..NVV_END 范围内 BREATHING token 的原始 logit 最高（预训练权重偏差）
- 即使原始 logit 差异很小，+4.0 后 softmax 将其推到接近 1.0

**设计修复 — Fix 3: NVV 后处理还原 + bias 策略调整**:

**3a. `ctc_prealign.py` 新增 NVV→拼音还原 Pass**:

在 `_normalize_english` 之后增加 `_reclaim_nvv_pinyin`:

```python
def _reclaim_nvv_pinyin(ctc_dir: Path, pinyin_dir: Path) -> int:
    """Revert NVV tokens that are actually misclassified pinyin syllables.
    
    A NVV token is likely pinyin if ALL of:
      - Word matches pinyin syllable pattern: [a-z]+[1-5]  (has tone digit)
      - Duration < 400ms (true NVV like BREATHING > 500ms)
      - Adjacent tokens are pinyin syllables (not English / other NVV)
      - Position in sentence is mid-sentence (not at a natural pause)
      - The .lab file has the original pinyin at this position
    """
    reverted = 0
    for tokens_path in sorted(ctc_dir.glob("*_tokens.jsonl")):
        stem = tokens_path.stem.replace("_tokens", "")
        tokens = [...]  # load
        lab_path = pinyin_dir / f"{stem}.lab"
        lab_tokens = lab_path.read_text().strip().split() if lab_path.exists() else []
        
        for i, tok in enumerate(tokens):
            if not is_nvv_token(tok["word"]):
                continue
            dur = tok["end_s"] - tok["start_s"]
            
            # Check 1: Duration fits pinyin syllable range
            if dur > 0.400:
                continue
            
            # Check 2: Look up original pinyin from .lab
            if i < len(lab_tokens):
                lab_tok = lab_tokens[i]
                if is_pinyin_syllable(lab_tok):
                    # Revert NVV → original pinyin
                    tok["word"] = lab_tok
                    reverted += 1
                    print(f"  [nvv_reclaim] {stem}: {tok['word']} ← NVV")
        
        # Write back...
    return reverted
```

**3b. (可选) `make_patched_inference` 中 NVV bias 条件化**:

当前 bias 对所有 blank 帧生效。改为仅对**连续 blank 帧**中段施加 bias:

```python
# Before: all blank frames get bias
x[is_blank, NVV_START:NVV_END + 1] += bias_value

# After: only sustained blank runs (≥3 consecutive blank frames ≈ 180ms)
blank_runs = []
j = 0
while j < len(is_blank):
    if is_blank[j]:
        s = j
        while j < len(is_blank) and is_blank[j]:
            j += 1
        if j - s >= 3:  # sustained silence
            blank_runs.append((s, j))
    else:
        j += 1

biased_mask = torch.zeros_like(is_blank)
for s, e in blank_runs:
    # Bias only middle frames (avoid edge frames near speech)
    mid_start = s + 1
    mid_end = e - 1
    if mid_end > mid_start:
        biased_mask[mid_start:mid_end] = True

x[biased_mask, NVV_START:NVV_END + 1] += bias_value
```

这确保 NVV bias 仅在持续静音段（≥180ms）生效，不在音节边界触发。

**涉及文件**: `scripts/ctc_prealign.py` (3a, 3b), `scripts/pipeline_utils.py` (无修改)

---

### 修改点汇总

| ID | 文件 | 函数 | 修改 | 优先级 |
|----|------|------|------|--------|
| **Fix-1** | `normalize_english_tokens.py` | `normalize_stem` (新增) | **Pass 2 碎片回收**: 合并残留单字母/短英文碎片到相邻英文词 | **P0** |
| **Fix-2a** | `ctc_prealign.py` | `make_patched_inference` | **参考文本保护**: ASR 英文词与 ref_text 不一致时回退到 ref | **P1** |
| **Fix-2b** | `normalize_english_tokens.py` | `_token_matches_ref` | **元音约束**: pinyin→English 仅当 ref ≥2 元音 (对齐 Case 10) | **P0** |
| **Fix-2c** | `normalize_english_tokens.py` | `normalize_stem` (新增) | **合并验证**: 合并后字母组成与参考词不一致时告警 | **P1** |
| **Fix-3a** | `ctc_prealign.py` | `_reclaim_nvv_pinyin` (新增) | **NVV 还原**: pinyin 模式 + <400ms → 还原为拼音 | **P0** |
| **Fix-3b** | `ctc_prealign.py` | `make_patched_inference` | **NVV bias 条件化**: 仅 ≥3 帧连续 blank 中段 bias | **P2** (可选优化) |

### 预期修复效果

| 问题样本 | 修复前 | Fix-1 | Fix-2a/b | Fix-3a | 修复后 |
|---------|--------|-------|----------|--------|--------|
| 000021 SOS+OS | `SOS`(180ms) + `OS`(420ms) | ✅→SOS(600ms) | — | — | `SOS` 合并 |
| 000135 f+an | `f`(60ms) + `an`(60ms) | ✅→fan(120ms) | — | — | `fan` 合并 |
| 000248 s+how | `s`(60ms) + `how`(360ms) | ✅→show(420ms) | — | — | `show` 合并 |
| 000352 f+an | `f`(60ms) + `an`(300ms) | ✅→fan(360ms) | — | — | `fan` 合并 |
| 000044 Instagram→sta | `Instagramsta` | — | ✅→Instagram | — | `Instagram` 修正 |
| 000021 li+ve→leave | `leave` | — | ✅→live | — | `live` 修正 |
| 000021 zan2→BREATH | `BREATHING`(240ms) | — | — | ✅→zan2 | `zan2` 还原 |
| 000248 jin1→BREATH | `BREATHING`(180ms) | — | — | ✅→jin1 | `jin1` 还原 |

### 验证方法

```python
# Fix-1 验证: 无残留短英文碎片
for tokens_path in ctc_dir.glob("*_tokens.jsonl"):
    tokens = json.loads(...)
    for i, t in enumerate(tokens):
        if is_english_token(t["word"]) and len(t["word"]) <= 2:
            dur = t["end_s"] - t["start_s"]
            if dur < 0.080:
                # Check if adjacent to longer English word
                if not ((i>0 and is_english_token(tokens[i-1]["word"]) and len(tokens[i-1]["word"])>=3)
                     or (i<len(tokens)-1 and is_english_token(tokens[i+1]["word"]) and len(tokens[i+1]["word"])>=3)):
                    print(f"  ORPHAN: {t['word']} {int(dur*1000)}ms in {tokens_path.stem}")

# Fix-2 验证: 英文词与原始 JSONL 一致
for stem, orig_text in original_jsonl_texts.items():
    tokens = load_tokens(stem)
    orig_eng = set(re.findall(r'[a-zA-Z]{2,}', orig_text))
    ctc_eng = set(t["word"] for t in tokens if is_english_token(t["word"]))
    if orig_eng != ctc_eng:
        print(f"  DRIFT: {stem}: {orig_eng} -> {ctc_eng}")

# Fix-3 验证: 无 pinyin 被误判为 NVV
for tokens_path in ctc_dir.glob("*_tokens.jsonl"):
    tokens = json.loads(...)
    for t in tokens:
        if is_nvv_token(t["word"]):
            dur = t["end_s"] - t["start_s"]
            if dur < 0.400:
                print(f"  NVV_SUSPECT: {t['word']} {int(dur*1000)}ms in {tokens_path.stem}")
```

---

### ria 名字完整性专项保护 (Fix-4)

**触发条件**: "ria" 是核心 VTuber 名字，必须保证在任何管线阶段都是完整小写 `ria`，不被碎片化或大小写不一致。

**当前 ria 处理链路及缺口**:

```
文本级: replace_ria_variants() → "瑞娅/瑞亚/瑞雅/瑞啊" → "ria"  ✅
                                                ↓
CTC 分词: NVASR tokenizer 对 OOV "ria" 的 3 种拆分:
  ├─ Pattern A: "rui4" + "ya4"  (中文音译)
  ├─ Pattern B: "R" + "I" + "A"  (单字母拆分)  
  └─ Pattern C: "R" + "ia"      (混合拆分)
                                                ↓
Token 合并: 单字母合并 (ctc_prealign.py:1325-1356)
  ├─ Pattern A: 不适用 (拼音音节非单字母)
  ├─ Pattern B: "R"+"I"+"A" → "RIA" ✅ (但大写!)
  └─ Pattern C: "R"+"ia" → "R"+"ia" ❌ 未合并 (ia 是2字符)
                                                ↓
_normalize_ria: regex 替换 (ctc_prealign.py:838-839)
  ├─ Pattern A: "rui4 ya4" → "ria" ✅
  ├─ Pattern B: "RIA" → 不匹配 ❌ (全大写不匹配 regex)
  └─ Pattern C: "R ia" → 不匹配 ❌
                                                ↓
_merge_ria_tokens: tokens.jsonl 合并 (ctc_prealign.py:795-796)
  ├─ Pattern A: "ruiN+yaN" → "ria" ✅
  ├─ Pattern A': "ruiN+aN" → 不合并 ❌ (仅检查 yaN!)
  ├─ Pattern B: "R"+"I"+"A" → 已合并不适用
  └─ Pattern C: 不适用
                                                ↓
normalize_en: 参考文本匹配
  ├─ Pattern B: "RIA"→"ria" ✅ (参考文本有 "ria")
  └─ Pattern C: "R"+"ia"→"ria" ✅ (碎片匹配合并)
```

**识别出的 3 个缺口**:

| Gap | 位置 | 现象 | 严重度 |
|-----|------|------|--------|
| **Gap-A** | `_merge_ria_tokens` line 795-796 | `ruiN + aN` → .lab 修复了但 tokens.jsonl **未合并** | **P0** |
| **Gap-B** | 单字母合并 line 1347-1348 | `"R"+"I"+"A"` → `"RIA"` 大写，依赖 normalize_en 修正大小写 | **P1** |
| **Gap-C** | normalize_en `_token_matches_ref` line 126 | `"ria"` 参考词可能被 pinyin→English 无条件匹配污染，导致碎片合入错误的目标词 | **P1** |

**设计修复 — Fix 4a: `_merge_ria_tokens` 增加 `ruiN + aN` 模式**:

`ctc_prealign.py` `_merge_ria_tokens`, line 795-796:
```python
# Before:
if (re.match(r'^rui[0-5]$', w) and i + 1 < len(entries)
        and re.match(r'^ya[0-5]$', entries[i + 1]["word"])):

# After:
if (re.match(r'^rui[0-5]$', w) and i + 1 < len(entries)
        and re.match(r'^(ya|a)[0-5]$', entries[i + 1]["word"])):
```

**设计修复 — Fix 4b: 单字母合并后强制 ria 小写**:

在单字母合并逻辑 (ctc_prealign.py line 1347-1348) 之后增加规范化:

```python
# After merged_pinyin.append({"word": "".join(letters), ...})
# NEW: Force lowercase for known proper names
_KNOWN_NAMES = frozenset({"ria", "noa", "mila"})
merged_word = "".join(letters)
if merged_word.lower() in _KNOWN_NAMES:
    merged_pinyin[-1]["word"] = merged_word.lower()
```

**设计修复 — Fix 4c: normalize_en 增加 ria 硬保护**:

在 `normalize_stem` (normalize_english_tokens.py) 中增加 ria 专项检查:

```python
# normalize_english_tokens.py normalize_stem, before changes loop (~line 271)
# NEW: Hard protection for "ria" — never merge into another word,
# never leave as fragments. Ria is always standalone lowercase.

_PROTECTED_NAMES = frozenset({"ria"})

for ri, indices in sorted(ref_to_lab.items()):
    en_word = en_ref_positions[ri]
    
    # [NEW] Protected name: ensure standalone and lowercase
    if en_word.lower() in _PROTECTED_NAMES:
        # 1. Gather all fragment indices that form "ria"
        # 2. Force lowercase in .lab and tokens
        # 3. Merge timestamps: start=earliest, end=latest
        # 4. Mark as handled, skip normal merge logic
        _merge_protected_name(ri, en_word.lower(), indices, ...)
        continue
```

**设计修复 — Fix 4d (推荐): 统一 ria 后处理校验函数**:

新增 `_protect_ria` 函数, 在 ctc_prealign.py 和 normalize_english_tokens.py 两处调用:

```python
# ctc_prealign.py / normalize_english_tokens.py 通用
def _protect_ria(tokens: list[dict], lab_tokens: list[str]) -> tuple[list, list]:
    """Ensure "ria" is always a complete, lowercase, standalone token.
    
    Handles all known fragmentation patterns:
    - "R"+"I"+"A" → "ria"  (single-letter split)
    - "R"+"ia"    → "ria"  (mixed split)
    - "RIA"       → "ria"  (case normalization)
    - "ruiN"+"yaN"→ "ria"  (pinyin phonetic, also covered by _normalize_ria)
    - "ruiN"+"aN" → "ria"  (pinyin variant, Gap-A)
    
    Returns (updated_tokens, updated_lab_tokens).
    """
    RIA_FRAGMENT_SETS = [
        frozenset({"r", "i", "a"}),      # single letters
        frozenset({"r", "ia"}),           # mixed
        frozenset({"ria"}),               # case fix only
        frozenset({"rui4", "ya4"}),       # pinyin phonetic
        frozenset({"rui2", "a1"}),        # pinyin variant
    ]
    
    i = 0
    new_tokens, new_lab = [], []
    while i < len(tokens):
        # Check if tokens[i:i+N] form "ria" fragments
        matched = None
        for n in range(1, 4):  # try 1-3 consecutive tokens
            if i + n > len(tokens):
                break
            fragment_set = frozenset(
                t["word"].lower() if isinstance(t, dict) else t.lower()
                for t in (tokens[i:i+n] if isinstance(tokens[0], dict) else lab_tokens[i:i+n])
            )
            if fragment_set in RIA_FRAGMENT_SETS:
                matched = n
                break
        
        if matched:
            # Merge fragments → single "ria" token
            if isinstance(tokens[0], dict):
                s = tokens[i]["start_s"]
                e = tokens[i + matched - 1]["end_s"]
                new_tokens.append({"word": "ria", "start_s": s, "end_s": e,
                                   "start_ms": round(s*1000), "end_ms": round(e*1000),
                                   "type": "word"})
            new_lab.append("ria")
            i += matched
        else:
            if isinstance(tokens[0], dict):
                new_tokens.append(tokens[i])
            new_lab.append(lab_tokens[i] if isinstance(lab_tokens, list) else tokens[i])
            i += 1
    
    return new_tokens, new_lab
```

**调用点**:
1. `ctc_prealign.py` `make_patched_inference` → 单字母合并之后调用 `_protect_ria(words_pinyin, lab_tokens)`
2. `normalize_english_tokens.py` `normalize_stem` → 合并循环之后调用 `_protect_ria(new_ctc, new_lab)`

**修改点汇总 — ria 专项**:

| ID | 文件 | 函数 | 修改 | 优先级 |
|----|------|------|------|--------|
| **Fix-4a** | `ctc_prealign.py` | `_merge_ria_tokens` | `ya[0-5]` → `(ya\|a)[0-5]` 扩展 tokens.jsonl 合并 | **P0** |
| **Fix-4b** | `ctc_prealign.py` | 单字母合并后 (line ~1356) | 已知名称 (ria/noa/mila) 强制小写 | **P0** |
| **Fix-4c** | `normalize_english_tokens.py` | `normalize_stem` | ria 硬保护: 独立 token + 小写 | **P1** |
| **Fix-4d** | `ctc_prealign.py` | `_protect_ria` (新增) | **推荐**: 统一 ria 后处理校验，所有 2 个调用点 | **P0** |

### 关联样本

- 测试集: `/tmp/test_en_ctc/ctc_pretg/` (10 条)
- 数据集: `/mnt/Raw/新版合成英文数据` (54000 条)

### 实施记录 (2026-08-04)

**已实施的修改**:

| ID | 文件 | 行号 | 修改 | 状态 |
|----|------|------|------|------|
| **Fix-4a** | `ctc_prealign.py` | ~796 | `ya[0-5]` → `(ya\|a)[0-5]` 扩展 `_merge_ria_tokens` | ✅ |
| **Fix-4b** | `ctc_prealign.py` | ~42-45, ~1357-1364 | `_SINGLE_LETTER_LOWERCASE_NAMES` 常量 + 单字母合并后强制小写 | ✅ |
| **Fix-4d** | `ctc_prealign.py` | ~824-880, ~1480 | `_protect_ria()` 函数 + 单字母合并/NVV去重后调用 | ✅ |
| **Fix-4d** | `normalize_english_tokens.py` | — | (ria 保护由 ctc_prealign 层覆盖, normalize_en 不再额外处理) | ✅ |
| **Fix-2b** | `normalize_english_tokens.py` | ~124-129 | `_token_matches_ref` 拼音→英文增加 ≥2 元音约束 | ✅ |
| **Fix-3a** | `ctc_prealign.py` | ~986-1050 | `_reclaim_nvv_pinyin()` 函数 (保留作为安全网) | ✅ |
| **Fix-3a** | `normalize_english_tokens.py` | ~289 | NVV token 排除出 `en_ref_positions` (防 normalize_en 把拼音合并到 NVV) | ✅ |
| **Fix-3a** | `normalize_english_tokens.py` | ~311-316 | NVV pre-reclaim: 短 NVV + 邻接拼音 → 还原为原始拼音 | ✅ |
| **Fix-1** | `normalize_english_tokens.py` | ~63-130, ~453-490 | `_reclaim_fragments()` Pass 2 + `normalize_stem` 末尾调用 | ✅ |
| **Fix-NEW** | `ctc_prealign.py` | ~154-168, ~228-232 | `make_patched_inference` 增加 `enable_nvv` 参数, False 时跳过 blank-frame NVV bias | ✅ |
| **Fix-NEW** | `ctc_prealign.py` | ~1089-1090 | `--no-nvv` CLI 开关 | ✅ |
| **Fix-NEW** | `ctc_prealign.py` | ~1145 | `--all-gpus` 子进程转发 `--no-nvv` | ✅ |
| **Fix-NEW** | `run_pipeline.py` | ~754-755 | `nvv_enabled: false` 配置 → `--no-nvv` 转发 | ✅ |
| **Fix-NEW** | `hecheng_english_mfa.yaml` | ~80 | `nvv_enabled: false` 配置项 | ✅ |

**实测验证 (10 条样本, `--no-nvv`)**:

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| NVV token 误判 | 2 个 (`zan2`/`jin1` → BREATHING) | **0** ✅ |
| 英文碎片残留 | "SOS"+"OS" 两个独立 token | "SOS" 合并为 600ms ✅ |
| ria 完整性 | 依赖 normalize_en 链式修正 | 单字母合并 + 小写 + `_protect_ria` ✅ |
| fragment reclaim | 无 | 2 文件各吸收 1 碎片 ✅ |
| `f`+`an` → `fan` | 未合并 | **仍残留** (短碎片相邻但无长英文词可吸收) ⚠ |

**已知剩余问题**:

1. `f`(60ms) + `an`(60ms) → 应合并为 `fan`, 但 `_reclaim_fragments` 当前仅在短碎片邻接**长英文词**时吸收。两短碎片相邻时不会互相合并。后续可在 `_reclaim_fragments` 中增加"短碎片互相合并"分支。

2. `_reclaim_nvv_pinyin` 函数逻辑正确但时序敏感 — 在 pipeline 中运行需确保 `.lab` 文件未被 `_normalize_english` 污染。当前通过在 `normalize_english_tokens.py` 内部做 NVV pre-reclaim 来绕过此问题。

**`--no-nvv` 方案说明**:

`--no-nvv` 是解决 NVV 误判的**首选方案**。设置后 `make_patched_inference` 跳过 blank-frame NVV bias, 模型完全不会检测 NVV token, 仅用 CTC 锚点给参考文本做时间戳。相比后处理还原 (`_reclaim_nvv_pinyin`), 这是从源头消除问题。

```bash
# 命令行
python ctc_prealign.py ... --no-nvv

# 配置文件
ctc_prealign:
  nvv_enabled: false
```

---

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
