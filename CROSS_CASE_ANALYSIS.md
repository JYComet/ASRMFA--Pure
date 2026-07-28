# 跨 Case 综合分析：共同根因与连锁问题

> 分析范围：`REGRESSION_ARCHIVE.md` Cases 1–13, `hanzi-tier-bugfix.md`, `FILTER_ANALYSIS_REPORT.md`
> 分析日期：2026-07-28

---

## 一、案例全景矩阵

| Case | 触发词 | 直接现象 | 一级根因 | 所属类别 |
|------|--------|---------|---------|---------|
| 1 | jie2 | 尾静音被 snap 回词 | Rule 3 误判 + 中间点未保护 | 边界冲突 |
| 2 | er4 | zhong3 过长/er4 过短 | Phase 1 merge vs Phase 3 snap 顺位冲突 | 阶段冲突 |
| 3 | le5 | 跨词界 eps 漏检 | `xmax <= ctc_end` 边界条件过严 | 条件守卫 |
| 4 | ti2 | 前词被短 NVV 裁短 | Rule 1 无条件 use_mfa=False | NVV 处理 |
| 5 | na2 | 词首过晚 | MFA eps 吞声母 → 词界偏移 | MFA 边界 |
| 6 | ji2 | 静音延伸被回截 | `is_punct(<eps>)` 误判 + end-trim 冲突 | 修复交互 |
| 7 | ru2 | pinyin_phones 词首间隙 | Phase 3.B 后 pinyin_phones 未同步 | 数据不同步 |
| 8 | \<sp0\> | sp0 残留 → mid_sp | `has_punct` 短路跳过 sp0 合并 | 条件守卫 |
| 9 | LAUGHTER | NVV 后静音链孤悬 | MFA 无法声学建模 NVV/标点 | NVV 处理 |
| 10 | qie4/切 | hanzi tier 拼音残留 | NW 对齐 + `_word_matches` 过度宽松 | 文本映射 |
| 11 | — | cjk_mismatch 未重定向 | 检查在路径决策之后执行 | 时序错误 |
| 12 | xia4/ta1 | 倒置 interval → 词丢失 | overlap 修复未设 word_end | 边界冲突 |
| 13 | laughter | pinyin_phones 小写 NVV | MFA 小写输出 + regex 大小写敏感 | 格式一致 |

---

## 二、六大共同根因

### 根因 1：MFA 声学模型的系统性盲区（Cases 1, 4, 9, 13）

MFA 是音素级强制对齐器，其声学模型在以下场景中**完全无法产生可靠输出**：

| 无法建模的 token 类型 | MFA 行为 | 引发的 Case |
|---------------------|---------|------------|
| **NVV** (`<LAUGHTER>`, `<BREATHING>`, …) | 无 phone 序列 → 占位符保留 → 边界随机浮动 | 4, 9, 13 |
| **英文词** | 有 IPA 词典但 phone 序列可能不完整 | 4 (English 分支) |
| **标点**（`，`, `。`, `！`, …） | 时长压缩到接近 0 | 1, 9 |
| **不送气声母/轻声** (b/d/g/de5/le5) | MFA 将低能量音素边界推入静音 | FILTER (word_in_silence) |

**连锁效应**：同一个 NVV token 需要经过至少 4 个处理阶段才能正确输出：
1. CTC 预对齐 → 给初略边界
2. MFA 对齐 → 占位符保留，边界可能不准
3. `_snap_to_ctc` → 边界修正（Case 4）
4. `absorb_nvv_trailing` → 尾部清理（Case 9）
5. `_finalize_textgrid` → 格式规范化（Case 13）

6 个 Case 中有 4 个与 **NVV 或标点被 MFA 丢弃/压缩**有关。

### 根因 2：CTC 锚点 vs MFA 边界 — 双权威冲突（Cases 1, 2, 4, 5, 12）

Pipeline 架构中同时存在两个边界权威：

```
CTC 锚点 (来自 NVASR)  ←→  MFA 边界 (来自声学模型对齐)
```

`_snap_to_ctc` 的 Rule 1-3 试图调和两者，但调和逻辑本身反复出 bug：

| Case | 冲突模式 | 调和失败原因 |
|------|---------|------------|
| 1 | CTC 时长 ≫ MFA 时长 (Rule 3) | 尾静音场景下应信任 MFA，但被比例检查否决 |
| 2 | Phase 1 修改了边界 → Rule 3 基于过期数据 | 两个阶段对同一 interval 做矛盾操作 |
| 4 | CTC 锚点本身就是噪声（短 NVV） | Rule 1 无条件否决 MFA |
| 5 | MFA eps 吞声母 → 词界偏移 | CTC 锚点偏早，能量谷底介于两者之间 |
| 12 | CTC snap 拉长前词 → 覆盖后词 | overlap 修复只设 start 不设 end |

**本质矛盾**：当 CTC 和 MFA 分歧时，没有**客观标准**判断谁更正确。当前方案是一个启发式规则的嵌套（比例、阈值、能量分析），每个规则都有其盲区。

### 根因 3：Pipeline 阶段间的数据不一致（Cases 2, 7, 11, 12）

多个阶段读写同一份 `words_tier` / `phones_tier`，但下游阶段不知道上游的修改：

```
Phase 1 (merge_short_silences)     → 修改 words 边界
Phase 3.A (_snap_to_ctc)           → 修改 words 边界（基于 Phase 1 的结果）
Phase 3.B (_refine_boundaries)     → 修改 words 边界
Phase 3.C (_inject_punctuation)    → 注入标点 interval
Phase 4 (absorb_*)                 → 吸收静音
Phase 5 (rebuild hanzi/pp)         → 基于最终 words 重建
```

| Case | 不一致类型 | 后果 |
|------|----------|------|
| 2 | Phase 1 合并了 zhong3+eps → Phase 3 的 Rule 3 基于已合并的边界计算 | er4 被压成 50ms |
| 7 | Phase 3.B 调整 words → pinyin_phones 未同步 | 词首 5ms 间隙 |
| 11 | cjk_mismatch 检查在路径决策之后 | 文件未被重定向 |
| 12 | overlap 修复只设 word_start → word_end 未更新 | 倒置 interval |

**修复模式**：每次都在事后增加同步步骤（修改 R, S, T, AA1），但没有架构层面的保证。

### 根因 4：条件守卫的边界值不精确（Cases 3, 6, 8, 10, 12）

多个修复的核心只是一行条件的微调：

| Case | 旧条件 | 新条件 | 影响 |
|------|--------|--------|------|
| 3 | `xmax <= ctc_end + 0.05` | `xmin < ctc_end + 0.05` | 跨词界 eps 从漏检变检出 |
| 6 | `is_punct(<eps>)` | `is_silence` 先检查 | eps 从"标点"变"静音" |
| 8 | `if sil_label is None or has_punct: continue` | 区分 sp0 vs sp1-3 | sp0 从被跳变被合并 |
| 10 | `return True`（pinyin→English） | `vowel_count >= 2` | OH/OP 不再被误匹配 |
| 12 | `m[0] < punct_start` | `m[1] <= punct_start + 0.001` | 尾静音从保留变吸收 |

**模式**：这些不是逻辑错误，而是**阈值/守卫条件未考虑到边界场景**。说明初始设计时对边缘情况的测试覆盖不足。

### 根因 5：静默丢弃与静默降级（Cases 6, 8, 9, 12）

多个 Case 的共同特征是一个中间步骤**静默地**丢弃或修改了 interval，导致下游看到不一致的状态：

| Case | 静默操作 | 下游后果 |
|------|---------|---------|
| 6 | end-trimming 截回延伸后的静音 | ji2 尾静音丢失 |
| 8 | `has_punct → continue` 跳过 sp0 合并 | sp0 残留 → mid_sp 误报 |
| 9 | 无代码吸收 NVV 后的静音链 | sp2 孤悬 → mid_sp 误报 |
| 12 | `if e > s` 过滤丢弃倒置 interval | ta1 消失，hanzi tier 缺字 |

**模式**：中间步骤不做显式报错/警告，只在最终 QC 阶段（mid_sp, cjk_mismatch）才被发现——而且 Case 11 暴露了最终 QC 本身也有时序 bug。

### 根因 6：上游格式假设被下游打破（Cases 10, 13）

一个模块对输入格式的假设，被上游模块的实际输出打破：

| Case | 假设 | 实际 | 打破来源 |
|------|------|------|---------|
| 10 | pinyin 不会精确匹配短英文（OH/OP） | `_word_matches` 无条件返回 True | pypinyin 多音字默认声调 |
| 13 | NVV 始终大写 | MFA 输出小写 `laughter` | MFA OOV 处理逻辑 |

---

## 三、Case 间直接因果链

### 链 1：NVV 处理链（Cases 4 → 9 → 13）

```
Case 4: NVV 短时长边界修复
  └→ 但 NVV 后的标点+静音链未被处理
      └→ Case 9: absorb_nvv_trailing 新增
          └→ 但 MFA 输出小写 NVV 仍不被识别
              └→ Case 13: regex IGNORECASE + 规范化
```

**当前残留风险**：`absorb_nvv_trailing` 的 NVV 检测依赖 `is_nvv_token`（检测 `NVV_NAMES` 中的大写名称）。如果 MFA 仍输出小写且 `is_nvv_token` 大小写敏感，则小写 NVV 后的标点+静音链可能再次残留。

### 链 2：边界调整冲突链（Cases 1 → 2 → 12）

```
Case 1: Rule 3 比例检查 → ratio_skip 修复
  └→ Case 2: Phase 1 合并后 Rule 3 基于过期数据 → prev_was_silence_extended 修复
      └→ Case 12: overlap 修复只设 start 不设 end → 倒置保护修复
```

**当前残留风险**：`prev_was_silence_extended` (修改 E) 和 overlap 倒置保护 (修改 AC) 的交互。当 `prev_was_silence_extended = True` 时缩短前词（Case 2），但如果前词就是被 CTC snap 拉长的（Case 12 pattern），缩短前词可能导致它又变回 MFA 边界——形成振荡。

### 链 3：静音合并/吸收链（Cases 3 → 6 → 8 → 9）

```
Case 3: 跨词界 eps 检测 → xmin 条件修复
  └→ Case 6: eps 被误判为 punct → is_silence 优先检查
      └→ Case 8: sp0 被 has_punct 跳过 → 无条件合并
          └→ Case 9: sp1-3 未被标点吸收 → absorb_silence_into_punct
```

**当前残留风险**：Case 6 的 `is_silence` 优先检查是否在所有使用 `is_punct` 的地方都正确应用？如果一个新场景把 `<eps>` 当作标点处理，Case 6 的修复可能不覆盖。

### 链 4：文本轨一致性链（Cases 7 → 10 → 11）

```
Case 7: pinyin_phones 与 words 不同步 → 同步修复 (R, S, T)
  └→ Case 10: hanzi tier 拼音残留 → 重写映射逻辑 (X, Y, Z)
      └→ Case 11: cjk_mismatch 时序 bug → 检查前置 (AA1, AA2)
```

**当前残留风险**：Case 10 的 `_build_hanzi_tier` 新逻辑和 `_normalize_word_spellings` 旧逻辑（仍用 NW 对齐）可能对同一 segment 产生不同的 label 判断。

---

## 四、可能存在但尚未发现的连锁问题

### 4.1 `_normalize_word_spellings` NW 对齐残留（已验证：误报）

~~**背景**：Case 10 修复了 `_build_hanzi_tier`（改用顺序映射），但 `_normalize_word_spellings` 仍使用 NW 对齐 + `_word_matches`。虽然 `_word_matches` 的 vowel 约束修复了（修改 X），但 NW gap-first 回溯仍可能在特定场景下产生次优对齐。~~

**验证**（2026-07-28）：用真实代码测试了 6 个场景（包括原始 bug doc 场景、pinyin 与 ≥2 vowel 英文相邻、交错 CJK+EN），全部正确。CJK 精确匹配（cost 0）天然优先于 pinyin→English 模糊匹配（cost 0 但 gap 操作增加总代价）。唯一 "misalignment" 是 `rui4↔ria` 的拼音渲染场景，这是**预期行为**。

**结论**：此风险不存在。vowel 约束 + CJK 精确匹配的组合已正确覆盖所有场景。

### 4.2 `is_nvv_token` 大小写敏感性（已确认安全）

~~**背景**：Case 13 修复了 `_NVV_PATTERN` regex 的大小写，但 `is_nvv_token` 函数本身可能仍然大小写敏感。~~

**核实**：`is_nvv_token` 内部已做 `token.strip().strip('<>').upper()`，大小写不敏感。Case 9 的 `absorb_nvv_trailing` 依赖 `is_nvv_token`，不会因大小写遗漏 NVV。**此风险不存在。**

### 4.3 `_refine_boundaries_by_energy` 与 Case 12 overlap 保护的交互（中风险）

**背景**：Case 12 的修改 AC 在 `_snap_to_ctc` 中增加倒置保护（`word_end = word_start + max(mfa_dur, 0.030)`）。但 `_refine_boundaries_by_energy`（Phase 3.B）在 snap 之后运行，可以再次调整边界。

**潜在场景**：snap 阶段用修改 AC 扩展了 word_end 到一个非自然的点（word_start + 30ms），然后 energy refinement 可能基于这个非自然边界做进一步的延伸/截断。

**风险等级**：低。30ms 是保守值，且 energy refinement 有独立阈值。

### 4.4 `_inject_punctuation` 静默丢弃的普适性（低风险，已分析）

**背景**：Case 12 发现 `_inject_punctuation` 用 `if e > s` 过滤倒置 interval。还发现多处用 `e > s + 0.001` 过滤近零时长 interval（line 2151, 2181, 2203, 2233, 2263）。

**核实**：
- `if e > s`（line 2105）：无容差。Case 12 的 AC 修改防止了倒置 interval 的产生，此过滤当前不会误杀合法 interval。但作为 defense-in-depth，建议改为 `e > s + 0.001` 或增加 warning。
- `e > s + 0.001`（5 处）：用于过滤合并后的近零时长残留，1ms 容差合理，正常词不会受影响。

**结论**：当前安全，但 line 2105 的零容差过滤缺少 warning，建议增加。

### 4.5 `_extract_word_chars` 特殊字符处理（→ 已确认 bug，已修复）

**背景**：`_extract_word_chars` 将所有非 alpha/digit/CJK 字符当作标点处理。

**验证**（2026-07-28）：
```
_extract_word_chars("<LAUGHTER>你好") → ['<', 'LAUGHTER', '>', '你', '好']  ✗
```

`<` 和 `>` 被当作标点拆分，NVV token `<LAUGHTER>` 变成 3 个独立单元。虽被 `_finalize_textgrid` 的 NVV 包裹逻辑掩盖（最终输出仍正确），但中间状态不正确：

- `ref_alpha` 收到 `"LAUGHTER"` 而非 `"<LAUGHTER>"`
- `_build_hanzi_tier` 将 hanzi label 设为 `"LAUGHTER"`（无括号），依赖 `_finalize_textgrid` 事后修复

**修复**（修改 Z2）：`_extract_word_chars` 中 `<` 和 `>` 特殊处理：
- `<` — flush 已有 buffer，开启新 group
- `>` — 加入 buffer 后立即 flush，确保后续字符独立
- 同时支持 `<sp1>`、`<sp2>` 等 silence tag 的正确分组

**影响**：此 bug 是 Case 13 (NVV 大小写不一致) 的同根问题——两者都涉及 NVV token 的文本处理不完整。Case 13 修复了大小写识别，本修复确保 bracket 不被拆分。

### 4.6 警告重复问题（低风险，已修复）

Case 10 补充修改：第一次 `_build_hanzi_tier` 不再传 warnings。已修复。

### 4.7 FILTER_ANALYSIS_REPORT 与 REGRESSION_ARCHIVE 的交叉

Cases 8, 9 的直接后果是 `mid_sp` 误报（48 个文件，占过滤的 1.7%）。FILTER_ANALYSIS_REPORT 在分析 `mid_sp` 时提出的解决方案（"sp0→前词合并"、"英文/NVV相邻静音处理"）已在 Cases 8, 9 中实现（修改 U, V, W1, W2）。

但 FILTER_ANALYSIS_REPORT 中 `mid_sp` 的 4 个子原因中，**子原因 (a) "长停顿无标点"** 和 **子原因 (c) "CTC 锚点错位"** 尚未在 REGRESSION_ARCHIVE 中有对应的系统性修复。

---

## 五、架构层面的系统性建议

### 5.1 引入 Tier 一致性断言（防御性）

在 `process_one` 的最终写入前，增加跨 tier 一致性检查：

```python
# 每个 CJK 词在 hanzi tier 中必须是 CJK 字符，不能是拼音
# 每个 NVV token 在所有 tier 中格式统一
# pinyin_phones 的首/末音素必须对齐 words 边界
```

Cases 7, 10, 11, 12, 13 都可以被这类检查在**发生时就发现**，而不是等到最终 QC 或用户报告。

### 5.2 建立 Phase 间数据变更的追踪机制

当 Phase N 修改 `words_tier` 时，标记受影响的 tier 为 dirty：
- `words_tier` 变更 → `hanzi_tier` dirty → 必须重建
- `words_tier` 变更 → `pinyin_phones` dirty → 必须重建

Cases 2, 7 的根本原因是下游不知道上游修改了数据。

### 5.3 CTC vs MFA 仲裁的客观标准

当前 `_snap_to_ctc` 的 Rule 1-3 是启发式的。建议引入**第三个仲裁源**——能量分析：
- 当 CTC 和 MFA 边界分歧 > 阈值时，用能量谷底/起振点作为 tiebreaker
- Case 5 的能量谷底检测可以推广到 snap 决策中

### 5.4 增加中间步骤的显式 warning 而非静默丢弃

将以下静默操作改为显式 warning：
- `if e > s` 过滤（`_inject_punctuation`）→ warning: "dropped inverted interval"
- `if sil_label is None or has_punct: continue` → warning: "skipped spN merge due to punct"
- `is_punct(<eps>)` → 改为先检查 `is_silence`

### 5.5 NVV 处理的统一抽象

NVV token 的处理分散在 6+ 个位置：
- `_NVV_PATTERN` regex（格式识别）
- `is_nvv_token`（类型判断）
- `_snap_to_ctc`（边界修正）
- `absorb_nvv_trailing`（尾部清理）
- `_finalize_textgrid`（格式规范化）
- `build_pinyin_phones_tier`（phone 生成）

考虑将 NVV 处理统一为一个 `normalize_nvv(text: str) -> str` 和 `process_nvv_boundary(...)` 对。

---

## 六、总结

| 维度 | 发现 |
|------|------|
| **直接因果链** | 6 条 Case 间因果链，4 条已闭合（修复到位），2 条仍有残留风险 |
| **共同根因** | 6 大类：MFA 盲区、CTC/MFA 冲突、阶段不同步、条件守卫不精、静默丢弃、格式假设 |
| **未发现的连锁问题** | 7 个潜在问题，1 个高风险（NW 对齐残留），3 个中风险，3 个低风险 |
| **架构建议** | 5 条系统性改进：一致性断言、变更追踪、仲裁标准、显式报错、NVV 统一抽象 |
