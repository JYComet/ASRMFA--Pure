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

`<sp0>` 与 `wei4` 同时开始 (6.660)，说明这是 MFA 的对齐 artifact——MFA 在帧精度边界处插入了与后词重叠的静音标签。15ms 的间隙没有任何语义意义，应该被吸收掉。

但 `handle_unexpected_silences` 未能合并它，最终被 `mid_sp` 检测抓到，文件被误滤。

### 根因链

1. **MFA 对齐 artifact**: MFA 在 `,` (6.630-6.660) 和 `wei4` (6.660-6.820) 之间插入 `<sp0>` [6.660-6.675]。注意 `<sp0>.xmin == wei4.xmin == 6.660`——这是 MFA 帧精度边界处的残余标签，不是真正的静音。

2. **tg_word_idx 过滤**: line 600-601 将标点从内容词列表中排除。`le5` 和 `wei4` 被识别为两个相邻内容词，中间的 `,` 和 `<sp0>` 构成间隙。

3. **gap_sil 正确捕获 `<sp0>`**: line 611-617 扫描 words tier 在 `le5` 和 `wei4` 之间的间隔，跳过 `,` (is_silence=False)，在 `<sp0>` 处命中 → `gap_sil[k] = "<sp0>"`。

4. **gap_punct 检测到标点**: line 621-624 在 pinyin tokens 中检测到逗号 → `gap_punct[k] = True`。

5. **has_punct 短路跳过 (line 637)**: 
   ```python
   if sil_label is None or has_punct:
       continue
   ```
   `has_punct=True` → 直接 `continue`，跳过所有静音处理。`<sp0>` 既不被合并，也不被标记为 unexpected_silence。它在 words tier 中残留。

6. **mid_sp 检测命中 (line 4205-4210)**: 最终过滤阶段扫描 words tier，发现中间有非空静音标签 → `filter_reasons.append("mid_sp")`，文件被误滤。

7. **line 704-707 的零时长清理无法救回**: 清理条件是 `duration > 0.001 or not text.strip()`。`<sp0>` 有非空文本且时长 15ms > 1ms → 不会被清理。

### 关键代码位置

| 位置 | 行号 | 作用 |
|------|------|------|
| `handle_unexpected_silences` L637 | 637 | `has_punct` 短路跳过——**根因** |
| `handle_unexpected_silences` L634-648 | 634-648 | 主循环：gap 遍历 + sil_label/has_punct 判断 |
| `handle_unexpected_silences` L704-707 | 704-707 | 零时长清理（1ms 阈值，无法清除非空的 15ms `<sp0>`） |
| `process_one` mid_sp 检测 | 4205-4210 | 检测 words 层中间残留静音 → `mid_sp` |

### 应修复的逻辑

当 `has_punct=True` 但间隙中同时存在 `<sp0>` 时，不应直接 `continue` 跳过。`<sp0>` 应被合并到相邻的标点间隔中（扩展标点的 `xmax`），而不是残留在 words tier 里。

修复方向：在 line 637 的 `continue` 之前，增加对 `sil_label == "<sp0>"` 的特判——将 `<sp0>` 吸收到前一个标点间隔（或后一个标点间隔）中，从 words/phonemes/pinyin_phones 三个 tier 中删除 `<sp0>`。

```
修复前: le5[6.540-6.630] ，[6.630-6.660] <sp0>[6.660-6.675] wei4[6.660-6.820]
修复后: le5[6.540-6.630] ，[6.630-6.675]              wei4[6.660-6.820]
```

`<sp0>` 的 15ms 被标点吸收，逗号 xmax 从 6.660 延伸到 6.675。

### 验证方法

```python
# 在标点 + <sp0> 共存的间隙中，<sp0> 必须被合并
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

### 待处理

- **37443 yu2** (Case 9): 目标 end=0.78s，当前 end=0.81s。yu2 尾部 (0.75-0.78) 有明显能量衰减，但 gang1 辅音起振 (0.795s, RMS 0.022) 落在 yu2 边界内，tail_rms gate 阻止了裁剪。end-trimming 无法区分"本词元音衰减"和"后词辅音起振"。需多词边界检测或能量谷底分割。

### 已解决

- **37435 jiu4**：修改 P 修复。jiu4 [10.39-10.49]，目标 [10.38-10.50]。
- **37434 ru2**：silence-adjacent 词首前拉 (N)。冒号 [6.80-7.355]，ru2 [7.355-7.50]。
- **冒号 `：` 在白名单**：`_NORM_ALLOWED_PUNCT` 包含 `：`，会被保留

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
