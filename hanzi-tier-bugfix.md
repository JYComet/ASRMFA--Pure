# Hanzi Tier Bug：拼音残留为 qie4 的完整分析与修复

## 1. 问题现象

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

## 2. 完整链路追踪

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
  ④ _word_matches() 第 795 行过度宽松匹配
     qie4 是拼音音节格式 → 匹配任意英文单词（代价 0）
     qie4 ↔ "OH" 代价 0；qie4 ↔ "切" 代价 0
     NW gap-first 回溯：优先消费 "OH"，把 qie4 配对给 OH
     "切" 变 reference-only gap → 被跳过
              │
              ▼
  ⑤ hanzi tier: SURPRISE | qie4 | 片 | OP
     "切" 丢失，"qie4" 作为 CTC-only 标签残留
```

## 3. 完整的触发场景分类

### 3.1 场景 A：多音字默认声调错误（本次报告的核心场景）

**触发条件**：参考文本包含多音字，且 `pypinyin.lazy_pinyin()` 逐字模式下选择了错误的默认声调。

**典型例子**：
- `切`：在"切片"中应读 qiē (qie1)，但 `lazy_pinyin("切")` 返回 `qie4`
- `的`：在"目的"中应读 dì (di4)，但 `lazy_pinyin("的")` 返回 `de5`
- `了`：在"了结"中应读 liǎo (liao3)，但 `lazy_pinyin("了")` 返回 `le5`
- `为`：在"因为"中应读 wèi (wei4)，但 `lazy_pinyin("为")` 返回 `wei2`

**说明**：声调错误本身不会直接导致 hanzi tier 出问题（因为 `_word_matches` 用同样的 `lazy_pinyin` 做反向匹配，两边声调一致仍然能匹配上）。但它是加上场景 B 的**前提条件**——如果声调正确，即使规则过度宽松，精确匹配的代价 0 也能和宽松匹配拉开差距。

### 3.2 场景 B：拼音音节 + 短英文词相邻（直接触发条件）

**触发条件**：MFA words tier 中同时存在：
1. 拼音音节 token（如 `qie4`、`pian4`）
2. 短英文 token（如 `OH`、`OP`、`in`、`up`）
3. 两者在序列中位置相邻

**触发机制**：`_word_matches()` 第 795 行将任意拼音音节匹配到任意英文单词（代价 0），NW 全局对齐的 gap-first 回溯在代价相同时优先匹配英文词，导致 CJK 字被跳过。

**具体对齐路径对比**：

```
NW 输入:
  ctc: ["SURPRISE-OH", "qie4", "pian4", "OP"]
  ref: ["SURPRISE",  "OH",   "切",    "片",   "OP"]

旧 _word_matches 代价矩阵:
                SURPRISE  OH  切  片  OP
  SURPRISE-OH      0      0   1   1   1
  qie4             0      0   0   1   0   ← qie4↔OH 代价 0！
  pian4            0      0   1   0   0
  OP               0      1   1   1   0

回溯结果（gap-first tie-breaking）:
  (0,0), (1,1), (None,2), (2,3), (3,4)
   ↑         ↑       
   qie4↔OH   "切"变 gap 被跳过

hanzi tier: SURPRISE | qie4 | 片 | OP
                       ✗ 残留拼音
```

### 3.3 场景 C：MFA 合并英文相邻词

**触发条件**：MFA 将两个相邻英文词合并为一个 token（如 `SURPRISE` + `OH` → `SURPRISE-OH`）。

**影响**：增加了 token 序列的不确定性，使 NW 对齐更容易出错。修复后通过贪心子串匹配处理：一个 MFA token 可消费多个 alpha 参考单元。

### 3.4 场景 D：MFA 丢弃 token（极端情况，新增检测）

**触发条件**：上游 pipeline 阶段（normalize、CTC prealign）丢弃了某个 CJK 字符对应的拼音 token。

**影响**：顺序映射会导致下游所有 CJK 字符错位。新增的防御性检查会在此情况下输出 warning。

## 4. 修复方案

### 4.1 `_build_hanzi_tier()` 重写 — 顺序映射替代字典反向查找

**核心思想**：用户建议的"直接用 ASR 输出的汉字文本，按顺序一一对应 MFA 拼音词位置"。

```
CJK 字符：拼音音节按顺序消费参考文本中的下一个 CJK 字
  不依赖 pypinyin 声调准确性
  不依赖任何字典
  qie4 → 下一个未使用的 CJK 字 → "切" ✓

英文/NVV：贪心子串匹配
  SURPRISE-OH → 消费 "SURPRISE" + "OH" 两个参考单元
  "li" → 匹配 "live"（子串）
```

**关键改动**：[postprocess_textgrids.py:858-970](chinese_mfa_pipeline/scripts/postprocess_textgrids.py#L858)

- CJK 字符和英文词**分开处理**，使用独立的消费游标
- CJK：`is_pinyin_syllable(token)` → 从 `ref_cjk` 队列消费下一个字符
- 英文/NVV：`_alpha_text_matches(token, ref)` → 贪心消费匹配的 alpha 参考单元
- 标点：`is_punct()` → 原样透传，不消耗任何游标
- silence：保持 silence label 透传

### 4.2 `_word_matches()` 收紧 — 元音数量约束

**改动**：[postprocess_textgrids.py:788-798](chinese_mfa_pipeline/scripts/postprocess_textgrids.py#L788)

```python
# 旧：return True（任意拼音音节匹配任意英文词）
# 新：仅当英文参考词 ≥2 个元音字母时才匹配
vowel_count = sum(1 for ch in r if ch in 'aeiou')
return vowel_count >= 2
```

**效果**：
- `qie4` 不再匹配 `OH`（1 个元音）✓
- `qie4` 不再匹配 `OP`（1 个元音）✓
- `ai4` 仍然匹配 `idol`（2 个元音）✓（拼音渲染场景）
- `rui4` 仍然匹配 `ria`（2 个元音）✓（拼音渲染场景）

**注意**：此函数仍被 `_normalize_word_spellings()` 通过 `_align_word_sequences()` 使用，修复同时保护了该函数。

### 4.3 防御性数量不匹配检测（新增）

**位置**：[postprocess_textgrids.py:962-979](chinese_mfa_pipeline/scripts/postprocess_textgrids.py#L962)

当 words tier 中的拼音音节数量 ≠ 参考文本中的 CJK 字符数量时，向 `report["warnings"]` 输出包含具体原因的 warning：

```
# 拼音 token 多于 CJK 字符（token 冗余）
"hanzi tier mismatch: 4 pinyin tokens vs 2 reference CJK chars
 — 2 pinyin token(s) fell back (no more CJK chars to consume)"

# CJK 字符多于拼音 token（字符丢失）
"hanzi tier mismatch: 2 pinyin tokens vs 3 reference CJK chars
 — 1 reference CJK char(s) were not assigned to any pinyin token"
```

## 5. 核心注意点

### 5.1 `lazy_pinyin` 逐字模式的固有限制

`pypinyin.lazy_pinyin(ch)` 对每个字独立判断，无法利用上下文消歧。这是 pypinyin 库的固有限制。解决方案不是让 pypinyin 更准确（那需要 `pypinyin.pinyin()` 的 heteronym 模式，但也会出错），而是**让 hanzi tier 构建不依赖 pypinyin 的返回值**。

### 5.2 顺序映射的前提假设

顺序映射正确的前提是：**MFA words tier 中拼音音节的顺序和数量 = 参考文本中 CJK 字符的顺序和数量**。

在正常 pipeline 中这个前提始终成立：
- `chars_and_pinyin()` 按字符顺序生成 `.lab`：1 个 CJK 字符 → 1 个拼音音节
- `.lab` 经过 normalize 后送入 MFA，MFA 不会删除或重排 token
- MFA words tier 的拼音 token 数量 = 原始 CJK 字符数量

如果这个前提被破坏（token 在中间环节被丢弃），防御性检查会发出 warning。

### 5.3 标点处理

标点在 words tier 中作为独立 interval 存在时，`is_punct()` 检测后**原样透传，不消耗 CJK 或 alpha 游标**。标点被 MFA 挤掉（不在 words tier 中）时，对 CJK 顺序映射**无影响**——因为参考文本中的标点被 `is_word_like()` 过滤，从未进入 CJK 消费队列。

### 5.4 英文词合并与拆分

英文词的 token 边界可能与参考文本不一致：
- **合并**：`SURPRISE-OH`（一个 token 对应两个参考词）→ 贪心匹配依次消费
- **拆分**：`li` `ve`（两个 token 对应一个参考词 "live"）→ 第一个 token 匹配消耗参考词，第二个变 CTC-only gap

这些情况下，hanzi tier 的英文 label 与 words tier 的 token 不完全 1:1 对应，但 CJK 部分不受影响。

### 5.5 NW 对齐仍然保留

`_align_word_sequences()` + `_word_matches()` 仍被 `_normalize_word_spellings()` 使用。修复只改了 `_word_matches` 中的一条规则（元音约束），保留了其他所有匹配逻辑。`_build_hanzi_tier()` 不再使用 NW 对齐。

## 6. 修改文件清单

| 文件 | 行号 | 改动 |
|------|------|------|
| [postprocess_textgrids.py](chinese_mfa_pipeline/scripts/postprocess_textgrids.py) | 646-647 | `_finalise_textgrid` 新增 `warnings` 参数 |
| [postprocess_textgrids.py](chinese_mfa_pipeline/scripts/postprocess_textgrids.py) | 680 | 向 `_build_hanzi_tier` 传递 warnings |
| [postprocess_textgrids.py](chinese_mfa_pipeline/scripts/postprocess_textgrids.py) | 788-798 | `_word_matches` 拼音→英文匹配增加元音约束 |
| [postprocess_textgrids.py](chinese_mfa_pipeline/scripts/postprocess_textgrids.py) | 858-970 | `_build_hanzi_tier` 重写：顺序 CJK 映射 + 贪心 alpha 匹配 + 防御检查 |
| [postprocess_textgrids.py](chinese_mfa_pipeline/scripts/postprocess_textgrids.py) | 2811 | `process_one` 向 `_finalise_textgrid` 传递 `report["warnings"]` |
| [postprocess_textgrids.py](chinese_mfa_pipeline/scripts/postprocess_textgrids.py) | 3025 | `process_one` 向第二次 `_build_hanzi_tier` 调用传递 `report["warnings"]` |
