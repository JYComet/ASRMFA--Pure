# 新版合成英文数据对齐 — 问题分析与修复方案

> **数据源**: `\\RS3621\Research_TTS\Data\Raw\新版合成英文数据对齐`
> **样本**: 400 文件 (5668 总) + 300 音频文件
> **配置**: `configs/hecheng_english_mfa.yaml`
> **分析日期**: 2026-08-05

---

## 目录

1. [Issue ① — 文本重叠 (11%, 44 files)](#issue-1)
2. [Issue ② — 英文词 phone 边界偏移 (80 处)](#issue-2)
3. [Issue ③ — 英文词音素不足 (~35%)](#issue-3)
4. [Issue ④ — 异常长词 (1 case, le5=5.6s)](#issue-4)
5. [Issue ⑤ — Words/Hanzi 轨道间隙 (18-48%)](#issue-5)
6. [Issue ⑥ — pinyin_phones 轨道不连续 (27%)](#issue-6)
7. [汇总修复优先级](#summary)

---

## <a name="issue-1"></a>Issue ① — 文本重叠 (11%, 44 files)

### 现象

```
Instagrams[5.862-6.762] ↔ 上[6.760-7.070]  重叠 2ms
R[6.522-6.882]         ↔ 的[6.880-6.980]  重叠 2ms
的[7.242-7.422]         ↔ 好[7.420-7.630]  重叠 2ms
```

中文词和英文 token 相邻时的边界重叠，均为 2ms 量级。

### 根因链

#### Step 1: English token 使用 CTC 边界

`_snap_to_ctc()` [postprocess_textgrids.py:3534-3535](scripts/postprocess_textgrids.py#L3534-L3535):
```python
elif is_english_token(mfa_iv.text):
    use_mfa = False
```
English token 强制使用 CTC 锚点边界（无 MFA 声学模型），而相邻中文词使用 MFA 边界。

#### Step 2: CTC 边界与 MFA 边界存在微小偏差

CTC (NVASR) 的帧移是 40ms (SenseVoice)，MFA 的帧移是 10ms。两者在边界上存在系统性精度差异，CTC 锚点可能略早于 MFA 词尾，产生 2ms 级微重叠。

#### Step 3: `_snap_to_ctc` 的重叠预防阈值过紧

[_snap_to_ctc 重叠预防代码](scripts/postprocess_textgrids.py#L3661):
```python
if word_start < prev_end - 0.002:   # 只修复 > 2ms 的重叠
```
2ms = 0.002s。当 `word_start = prev_end - 0.002`（恰好 2ms 重叠），条件 `< prev_end - 0.002` 为 **False**，重叠被漏掉。

#### Step 4: `_fix_overlapping_boundaries` 的 5ms floor 再次漏过

[_fix_overlapping_boundaries](scripts/postprocess_textgrids.py#L967-L968):
```python
overlap = cur.xmax - nxt.xmin
if overlap <= 0.005:          # 忽略 ≤ 5ms 重叠
    continue
```
2ms 重叠远小于 5ms，被直接跳过。

#### Step 5: English 侧的特殊性

[_fix_overlapping_boundaries](scripts/postprocess_textgrids.py#L974-L979) 对 content 的判断:
```python
cur_is_content = (cur_text and not is_punct(cur_text)
                  and not is_silence(cur_text)
                  and not is_nvv_token(cur_text))
```
English token 不在排除列表中（仅排除 NVV/silence/punct），所以当重叠 > 5ms 且 < 30ms 时理论上会被修复。但 2ms 重叠连 5ms floor 都过不了。

### 影响函数

| 函数 | 文件:行号 | 角色 |
|------|----------|------|
| `_snap_to_ctc` | postprocess_textgrids.py:3661 | 重叠检测阈值 2ms → 漏过 2ms 重叠 |
| `_fix_overlapping_boundaries` | postprocess_textgrids.py:968 | 5ms floor → 二次漏过 |

### 修复方案

**方案 A (推荐)**: 降低 `_snap_to_ctc` 的重叠容忍度至 0

```python
# postprocess_textgrids.py:3661 — 修改前:
if word_start < prev_end - 0.002:

# 修改后:
if word_start < prev_end - 0.0005:  # 0.5ms tolerance (sub-frame)
```

同时将 `_fix_overlapping_boundaries` 的 floor 从 5ms 降至 1ms:

```python
# postprocess_textgrids.py:968 — 修改前:
if overlap <= 0.005:

# 修改后:
if overlap <= 0.001:  # 1ms — only skip sub-frame rounding artifacts
```

**方案 B (更彻底)**: 在 `_snap_to_ctc` 末尾添加无条件连续化步骤，确保所有相邻 word interval 之间无重叠无间隙（tiny gap 已有吸收代码 line 3802-3807，但 tiny overlap 没有对应的吸收代码）。

```python
# 在 _snap_to_ctc line 3807 之后添加:
# Eliminate tiny overlaps between consecutive word intervals.
for k in range(len(new_word_ivs) - 1):
    cur = new_word_ivs[k]
    nxt = new_word_ivs[k + 1]
    overlap = cur[1] - nxt[0]
    if 0 < overlap <= 0.005:
        mid = (cur[1] + nxt[0]) / 2.0
        new_word_ivs[k] = (cur[0], mid, cur[2], cur[3])
        new_word_ivs[k + 1] = (mid, nxt[1], nxt[2], nxt[3])
```

---

## <a name="issue-2"></a>Issue ② — 英文词 phone 边界偏移 (80 处)

### 现象

英文 token 的 phone (pinyin_phones tier) 起点与 word (words tier) 起点不对齐，phone 边界存在系统性偏移。

### 根因链

#### Step 1: English MFA 在独立音频段上对齐

`align_english_mfa.py` 将英文词提取为独立音频段（带 50ms padding, [config line 106](configs/hecheng_english_mfa.yaml#L106)），在子音频段上运行 English MFA。

#### Step 2: 英文段起始时间与 words tier 边界不同

提取段有 `padding_ms=50`，段内 MFA 给出的 `word_start` 参考点是段的起始而非原始音频的绝对时间。虽然 `align_english_mfa.py` 会转换回绝对时间（`en_word_start` / `en_word_end`），但这个绝对时间可能因段边界的 padding 效应与 CTC-snapped 的 word 边界有偏移。

#### Step 3: `_apply_en_phones` 的比例映射假设

[_apply_en_phones](scripts/postprocess_textgrids.py#L3930-L3931):
```python
en_start = en_entry.get("en_word_start", en_entry["word_start"])
en_end = en_entry.get("en_word_end", en_entry["word_end"])
en_dur = en_end - en_start if en_end > en_start else word_dur
```

English MFA 的 phone 时间按 `(phone_time - en_start) / en_dur` 计算相对位置，再线性映射到 CTC-snapped 的 `[w_start, w_end]`。

**问题**: 线性映射假设 English MFA 段的时间范围和 words tier 的时间范围是相同的比例关系。但实际上：
- English MFA 对齐的是 padded 段内的词，词的首尾在段内的位置可能略有偏移
- CTC 锚点的边界来自 NVASR（40ms 帧移），与 English MFA（10ms 帧移）的精度不同
- 当 `en_word_start > word_start`（English MFA 认为词开始得更晚），比例映射会导致首音素被压缩

#### Step 4: `en_mfa_windows` 过滤加剧偏移

[build_pinyin_phones_tier](scripts/postprocess_textgrids.py#L564-L578) 在 Phase 5 重建时使用 `en_mfa_windows` 过滤 English phones:
```python
if word_phones and en_mfa_windows:
    wl = w_iv.text.strip().lower()
    if wl in en_mfa_windows:
        es, ee = en_mfa_windows[wl]
        word_phones = [
            (s, e, t) for s, e, t in word_phones
            if t.startswith(EN_PHONE_PREFIX)
            or (s >= es - 0.3 and e <= ee + 0.3 ...)
        ]
```

0.3s 的容差窗口很大，但 English MFA 的 word 边界和 CTC word 边界之间的偏移会传递到 phones 上，使首/尾音素的时间戳偏离 word 边界。

### 影响函数

| 函数 | 文件:行号 | 角色 |
|------|----------|------|
| `_apply_en_phones` | postprocess_textgrids.py:3930-3931 | 比例映射假设不精确 |
| `build_pinyin_phones_tier` | postprocess_textgrids.py:564-578 | en_mfa_windows 过滤 |
| `align_english_mfa.py::build_en_corpus` | align_english_mfa.py:196 | padding 引入偏移 |

### 修复方案

**方案**: 在 `_apply_en_phones` 的比例映射后，将首音素的 start snap 到 word start，尾音素的 end snap 到 word end（与中文 phone 处理方式一致，见 Phase 5 的 [line 4799-4802](scripts/postprocess_textgrids.py#L4799-L4802)）。

```python
# 在 _apply_en_phones 的每个 English word 处理末尾（line 4023 前）添加:
# Snap first phone start to word start, last phone end to word end
en_phones_for_word = [iv for iv in new_phone_ivs 
                      if w_start - 0.005 <= iv.xmin and iv.xmax <= w_end + 0.005]
if en_phones_for_word:
    if en_phones_for_word[0].xmin > w_start + 0.002:
        en_phones_for_word[0] = Interval(w_start, en_phones_for_word[0].xmax, 
                                          en_phones_for_word[0].text)
    if en_phones_for_word[-1].xmax < w_end - 0.002:
        en_phones_for_word[-1] = Interval(en_phones_for_word[-1].xmin, w_end,
                                           en_phones_for_word[-1].text)
```

---

## <a name="issue-3"></a>Issue ③ — 英文词音素不足 (~35%)

### 现象

| 词 | Dict 音素数 | 实际音素 | 占比 |
|----|-----------|---------|------|
| RIA | 3 | 1 (RIA 整词) | 131 例中 31 例不足 |
| BGM | 6 | 1-2 | 高频不足 |
| AI | 2 | 1 | 17/18 例 |

"pinyin_phones tier 中这些英文词变成了自引用整词标签而非拆分后的 ARPABET 音素——英文路径的 FULL_WORD_AS_PHONE 等价问题。"

### 根因链

这是本报告中最复杂的 issue。需要追踪整个 Phase 3.5 → Phase 5 的数据流。

#### Step 1: Phase 3.5 成功注入了 English phones

[process_one Phase 3.5](scripts/postprocess_textgrids.py#L4455-L4477):
```python
if en_data:
    _, phones_tier = _apply_en_phones(words_tier, phones_tier, en_data)
    # ...
    synced_pp = build_pinyin_phones_tier(phones_tier, ipa_to_pinyin,
                                          words_tier, pinyin_dict)
    # ⚠️ 注意: 这里没有传 en_mfa_windows!
```

`_apply_en_phones` 将 English MFA 音素注入到 **phones** tier 中（以 `en:AA1`, `en:B` 等形式）。然后 `build_pinyin_phones_tier` 重建 pinyin_phones。

但 Phase 3.5 的 `build_pinyin_phones_tier` 调用 **没有传 `en_mfa_windows`**！这意味着在 Phase 3.5，English phones 不会被 `en_mfa_windows` 过滤，它们应该能正常通过。

#### Step 2: Phase 4 修改了 word 边界

Phase 4 的多步处理（`absorb_nvv_trailing`, `absorb_silence_into_punct`, `strip_edge_punctuation`, `_extend_word_into_ellipsis` 等）会修改 words tier 的边界。English word 的边界也可能被调整。

#### Step 3: Phase 5 的 FINAL `build_pinyin_phones_tier` 使用 `en_mfa_windows` 过滤

[process_one Phase 5](scripts/postprocess_textgrids.py#L4776-L4780):
```python
synced_pp = build_pinyin_phones_tier(final_phones_tier, ipa_to_pinyin,
                                      final_words_tier, pinyin_dict,
                                      en_mfa_windows=en_mfa_windows)  # ← 这次传了!
```

在 `build_pinyin_phones_tier` 内，对于 English token：

[line 561-598](scripts/postprocess_textgrids.py#L561-L598):
```python
if is_english_token(w_iv.text):
    if word_phones and en_mfa_windows:
        wl = w_iv.text.strip().lower()
        if wl in en_mfa_windows:
            es, ee = en_mfa_windows[wl]
            word_phones = [
                (s, e, t) for s, e, t in word_phones
                if t.startswith(EN_PHONE_PREFIX)
                or (s >= es - 0.3 and e <= ee + 0.3
                    and not _looks_chinese_phone(t)
                    and not is_silence(t))
            ]
        else:
            word_phones = []  # ← 关键: 词不在 en_mfa_windows 中 → 清空!
    if word_phones:
        # 正常: 转换为 ARPABET labels
        ...
        continue
    # ← 到这里表示 word_phones 为空 → 自引用 fallback
    new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, w_iv.text))
```

#### Step 4: Phase 4 边界变更导致 `en_mfa_windows` 查找失败

`en_mfa_windows` 是在 [Phase 3.5 前构建](scripts/postprocess_textgrids.py#L4435-L4441)的：
```python
en_mfa_windows: dict[str, tuple[float, float]] = {}
if en_data:
    for entry in en_data:
        es = entry.get("en_word_start", entry["word_start"])
        ee = entry.get("en_word_end", entry["word_end"])
        en_mfa_windows[entry["word_text"].strip().lower()] = (es, ee)
```

但 `en_mfa_windows` 的 key 仅仅是 `word_text`（如 `"ria"`），不包含时间信息。当文件中同一个英文词出现多次时（如 `"RIA"` 出现 2 次），`en_mfa_windows` 只保留最后一次出现的窗口！

**这就是根因**: 同一文件中重复出现的英文词，只有最后一次的 English MFA 窗口被保留。前面的出现会在 Phase 5 的 `build_pinyin_phones_tier` 中被清空（`word_phones = []`），然后 fall through 到自引用标签。

#### Step 5: `en_mfa_windows` 时间窗口过期

Phase 3C (`_inject_punctuation`) 和 Phase 4 的多个步骤会修改 English word 的边界。但 `en_mfa_windows` 保存的是 Phase 3.5 之前的 `en_word_start`/`en_word_end`。当 Phase 4 改变了 English word 的边界后，`en_mfa_windows` 中的旧窗口可能不再包含 phone 的时间范围，导致 phones 被过滤掉。

### 影响函数

| 函数 | 文件:行号 | 角色 |
|------|----------|------|
| `build_pinyin_phones_tier` | postprocess_textgrids.py:561-598 | English phone 过滤 + fallback |
| `process_one` (Phase 3.5) | postprocess_textgrids.py:4472 | 漏传 `en_mfa_windows` |
| `process_one` (Phase 5) | postprocess_textgrids.py:4778-4780 | 传入已过期的 `en_mfa_windows` |
| `process_one` | postprocess_textgrids.py:4436-4441 | `en_mfa_windows` 按 word_text 覆盖（丢失重复词） |

### 修复方案

**修复 1 — `en_mfa_windows` key 改为包含时间信息**（修复重复词覆盖）:

```python
# postprocess_textgrids.py:4436-4441 — 修改前:
en_mfa_windows: dict[str, tuple[float, float]] = {}
if en_data:
    for entry in en_data:
        es = entry.get("en_word_start", entry["word_start"])
        ee = entry.get("en_word_end", entry["word_end"])
        en_mfa_windows[entry["word_text"].strip().lower()] = (es, ee)

# 修改后: key 包含起始时间，支持重复词
en_mfa_windows: dict[tuple[str, float], tuple[float, float]] = {}
if en_data:
    for entry in en_data:
        es = entry.get("en_word_start", entry["word_start"])
        ee = entry.get("en_word_end", entry["word_end"])
        key = (entry["word_text"].strip().lower(), round(es, 2))
        en_mfa_windows[key] = (es, ee)
```

同时更新 [build_pinyin_phones_tier line 566](scripts/postprocess_textgrids.py#L566) 的查找逻辑：
```python
# 修改前:
if wl in en_mfa_windows:
    es, ee = en_mfa_windows[wl]

# 修改后: 用词的起始时间 + 文本匹配
matched_window = None
for (key_wl, key_ts), (es, ee) in en_mfa_windows.items():
    if key_wl == wl and abs(key_ts - w_iv.xmin) < 0.5:
        matched_window = (es, ee)
        break
if matched_window:
    es, ee = matched_window
```

**修复 2 — Phase 5 放宽过滤，优先保留已有 phones**:

Phase 3.5 已经将 English MFA phones 正确写入了 phones tier。Phase 5 不应重新过滤它们。在 `build_pinyin_phones_tier` 中，对 `en:` 前缀的 phone 做无条件保留（它们是 `_apply_en_phones` 注入的，已经在正确的 word 范围内）：

```python
# build_pinyin_phones_tier:561-578 修改:
if is_english_token(w_iv.text):
    if word_phones:
        # 无条件保留 en: 前缀的 phones (_apply_en_phones 已确保正确)
        # 其他 phones 用 en_mfa_windows 过滤
        filtered = []
        for s, e, t in word_phones:
            if t.startswith(EN_PHONE_PREFIX):
                filtered.append((s, e, t))  # 无条件保留
            elif en_mfa_windows:
                # ... 现有的过滤逻辑 ...
            else:
                filtered.append((s, e, t))  # 无 windows 信息时保留
        word_phones = filtered
    # ... 后续处理 ...
```

**修复 3 — Phase 3.5 的 `build_pinyin_phones_tier` 也传入 `en_mfa_windows`**:

[process_one line 4472](scripts/postprocess_textgrids.py#L4472):
```python
# 修改前:
synced_pp = build_pinyin_phones_tier(phones_tier, ipa_to_pinyin,
                                      words_tier, pinyin_dict)

# 修改后:
synced_pp = build_pinyin_phones_tier(phones_tier, ipa_to_pinyin,
                                      words_tier, pinyin_dict,
                                      en_mfa_windows=en_mfa_windows)
```

---

## <a name="issue-4"></a>Issue ④ — 异常长词 (le5 = 5.6s)

### 现象

```
le5 [9.720-15.554] dur=5.6s
prev: hao3  [9.720-9.914]
next: 。    [15.554-15.987]
```

5.6 秒的 `le5` — 在 `hao3` 和 `。` 之间吞掉了大段静音或未识别的内容。

### 根因链

#### Step 1: CTC 锚点给 le5 分配了过长的区间

NVASR 的 CTC 解码可能在 `le5` 和 `。` 之间的静音段上无法确定边界，将大段静音归入了 `le5` 的 CTC span。

#### Step 2: `_snap_to_ctc` duration-ratio 规则触发 snap

[_snap_to_ctc](scripts/postprocess_textgrids.py#L3601-L3603):
```python
if use_mfa and not has_mfa_phone_evidence and not ratio_skip \
   and (mfa_dur > ctc_dur * 2.0 or ctc_dur > mfa_dur * 2.0):
    use_mfa = False
```

当 `ctc_dur = 5.6s` 而 `mfa_dur` 正常（~0.2s），ratio > 2x，触发 `use_mfa = False`，词边界被 snap 到 CTC 锚点。

#### Step 3: `ratio_skip` 未触发

[`ratio_skip` 的检测条件](scripts/postprocess_textgrids.py#L3568-L3600) 检查是否有 trailing silence 或 punctuation 导致的 CTC span 膨胀。在这个 case 中：
- `has_trailing_sil` 检查 — le5 后面可能没有紧邻的 `<eps>`（静音是连续的长段，MFA 可能将其标为单个长 sil）
- `ratio_skip` 的 gap_sil 检查 — 如果 MFA 将整段静音标为一个 sil（而不是 `<eps>` 夹在 le5 和 `。` 之间），这个检查会失败

#### Step 4: 无 word-too-long 检测

后处理管线有 `fix_short_words`（延长过短词），但没有对应的 `fix_long_words`（缩短过长词）。`detect_issues` 中有 `word_too_short` 检查但没有 `word_too_long` 检查。

### 影响函数

| 函数 | 文件:行号 | 角色 |
|------|----------|------|
| `_snap_to_ctc` (ratio_skip) | postprocess_textgrids.py:3568-3600 | 未能识别此 case |
| `_snap_to_ctc` (duration ratio) | postprocess_textgrids.py:3601-3603 | 触发 snap 到错误 CTC |
| `detect_issues` | postprocess_textgrids.py:2255 | 缺少 word_too_long 检测 |

### 修复方案

**修复 1 — 添加 `word_too_long` 检测**（类似于已有的 `word_too_short`）:

```python
# detect_issues, 在 word_too_short 检测附近添加:
if w.duration > 3.0 and not is_english_token(w.text) and not is_nvv_token(w.text):
    issues.append({"rule": "word_too_long", "text": w.text, 
                   "duration": round(w.duration, 4)})
```

**修复 2 — `_snap_to_ctc` 添加绝对时长保护**:

```python
# _snap_to_ctc, 在 duration-ratio 规则 (line 3601) 之前添加:
# Absolute duration guard: any Chinese word > 3s is an alignment error.
# CTC anchor is likely inflated by unlabeled silence/content.
CTC_MAX_DUR = 3.0
if use_mfa and ctc_dur > CTC_MAX_DUR and mfa_dur < 1.0:
    use_mfa = True  # Force MFA — CTC anchor is clearly wrong
    # But also search forward for the real energy boundary
```

**修复 3 — 扩展 `ratio_skip` 检测**:

在 `ratio_skip` 逻辑中增加对"长静音夹在中间"的检测（不仅检查 `<eps>`，也检查长时间 silence gap）：

```python
# 在 _snap_to_ctc ratio_skip 检测 (line 3568) 中添加:
# Pattern (c): Large silence gap between MFA word end and CTC end
if not ratio_skip and ctc_end > mfa_end + 0.5:
    # CTC spans a long gap beyond MFA end — MFA is more trustworthy
    ratio_skip = True
```

---

## <a name="issue-5"></a>Issue ⑤ — Words/Hanzi 轨道间隙 (18-48%)

### 现象

- Batch1 (直播流程): 33/188 文件 (18%) — 41 处间隙
- Batch2 (礼物互动): 86/180 文件 (48%) — 160 处间隙

间隙模式：
```
lao2 → lao2  15ms    ← 相同词之间的空洞
tui1 → le5   30ms    ← 不同词之间
idol → ...    5ms    ← 英文词后
```

words 和 hanzi 完全镜像 —— 证明同步正确但源头有洞。

### 根因链

#### Step 1: MFA 对齐的帧精度残余

MFA 的帧移为 10ms，词边界可能不会精确地在声学边界上。`_snap_to_ctc` 的 gap 吸收代码 [line 3802-3807](scripts/postprocess_textgrids.py#L3802-L3807):

```python
for k in range(len(new_word_ivs) - 1, 0, -1):
    cur = new_word_ivs[k]
    prev = new_word_ivs[k - 1]
    gap = cur[0] - prev[1]
    if 0 < gap <= 0.005 and prev[3] == "word":
        new_word_ivs[k - 1] = (prev[0], cur[0], prev[2], prev[3])
```

**仅吸收 ≤ 5ms 的 gap**。5-30ms 的 gap 被保留为 `<spN>` 标签。

#### Step 2: `_inject_punctuation` 的小间隙吸收有条件限制

[_inject_punctuation line 2565-2591](scripts/postprocess_textgrids.py#L2565-L2591) 将 `<spN>` 间隙吸收到邻近词中，但有以下限制：
- 间隙 > 500ms → 跳过
- 句首间隙 → 跳过
- NVV 前无条件合并，但标点前仅合并不超过 500ms
- 优先合并到后一词，不成再合到前一词

这个逻辑处理了标点邻接的间隙，但**词间无标点的纯间隙**（两个相邻 pinyin 词之间的 gap）没有对应的吸收逻辑。

#### Step 3: 不同内容类型的影响

Batch2 (礼物互动, 48%) 比 Batch1 (直播流程, 18%) 间隙率高 2.7 倍。Batch2 可能包含更多：
- 英文词后的间隙（English token 强制 CTC → 边界精度不如 MFA）
- 短词/短语之间的自然停顿
- 情绪标签 (NVV) 周围的清理残余

#### Step 4: 间隙在 hanzi 中的镜像传播

因为 `_build_hanzi_tier` 使用 `_align_word_sequences` 将 words tier 逐词映射为 CJK 字符，words 层的 gap 会在 hanzi 中创建对应的空 interval。这是正确的行为（证明同步没问题），但问题源头在 words tier。

### 影响函数

| 函数 | 文件:行号 | 角色 |
|------|----------|------|
| `_snap_to_ctc` (gap 吸收) | postprocess_textgrids.py:3802-3807 | 仅吸收 ≤ 5ms |
| `_inject_punctuation` | postprocess_textgrids.py:2565-2591 | 仅吸收标点邻接间隙 |
| `_refine_boundaries_by_energy` | postprocess_textgrids.py:2875 | 不处理纯词间间隙 |
| `_build_hanzi_tier` | postprocess_textgrids.py:1487 | 镜像传播 gap |

### 修复方案

**修复 1 — 提高 `_snap_to_ctc` 的 gap 吸收阈值**:

```python
# postprocess_textgrids.py:3806 — 修改前:
if 0 < gap <= 0.005 and prev[3] == "word":

# 修改后: 吸收 ≤ 30ms 的 gap (MFA 3 帧精度)
if 0 < gap <= 0.030 and prev[3] == "word":
```

30ms 对应 MFA 的 3 个帧（每帧 10ms），是声学边界模糊的合理范围。超过 30ms 的 gap 更可能是真实的短停顿。

**修复 2 — 添加通用的微间隙吸收步骤**:

在 Phase 4 末尾（`_fix_overlapping_boundaries` 之后）添加一个通用步骤，将所有 < 30ms 的非语义间隙吸收到前一词或后一词：

```python
def _absorb_tiny_gaps(words_tier: Tier, max_gap_s: float = 0.030) -> Tier:
    """Absorb sub-frame gaps between consecutive content words."""
    intervals = list(words_tier.intervals)
    for i in range(len(intervals) - 2, -1, -1):
        cur = intervals[i]
        nxt = intervals[i + 1]
        if is_silence(cur.text) and cur.duration < max_gap_s:
            # Tiny silence gap — absorb into neighboring word
            # Prefer absorbing into the longer word
            prev_word = intervals[i - 1] if i > 0 else None
            if prev_word and not is_silence(prev_word.text):
                intervals[i - 1] = Interval(prev_word.xmin, nxt.xmin, prev_word.text)
                del intervals[i]
            elif not is_silence(nxt.text):
                intervals[i + 1] = Interval(cur.xmin, nxt.xmax, nxt.text)
                del intervals[i]
    return Tier(words_tier.name, words_tier.xmin, words_tier.xmax, intervals)
```

---

## <a name="issue-6"></a>Issue ⑥ — pinyin_phones 轨道不连续 (27%)

### 现象

全量 6881 文件中 1839 个存在 pp 轨道间隙，分布：
```
zh→zh:     41  ← 中文词之间（继承自 words tier）
punct→zh:  13  ← 标点后到中文词
zh→en:     13  ← 中文到英文词过渡
zh→punct:  11  ← 中文词到标点
en→zh:     11  ← 英文到中文词
```

### 根因链

#### Step 1: 大部分是 words tier 间隙的连锁反应

当 words tier 有间隙（Issue ⑤），`build_pinyin_phones_tier` 在遍历 words tier 时会在间隙位置创建一个 silence label interval，但如果 phones tier 中对应位置没有 silence phone，pp tier 就会留下一个空洞。

#### Step 2: `build_pinyin_phones_tier` 的 silence 处理

[line 473-480](scripts/postprocess_textgrids.py#L473-L480):
```python
if is_silence(w_iv.text) or not word or word in ("", "<eps>"):
    dur_label = silence_label(w_iv.duration)
    new_intervals.append(Interval(w_iv.xmin, w_iv.xmax, dur_label))
    while phone_idx < len(mfa_phones) and mfa_phones[phone_idx].xmax <= w_iv.xmax + 0.001:
        phone_idx += 1
    continue
```

这里只插入 duration-based silence label，但不尝试从 phones tier 中拷贝实际的 silence phone。如果 phones tier 中没有对应的 sil/sp 区间，pp tier 会有一个独立的 silence label。

但实际上这应该不造成"不连续"——silence label 本身就是 pp interval。问题在于：当 words tier 的 gap 处于两个内容词之间，而 phones tier 的 sil/sp 可能和 words tier 的 `<spN>` 不在同一时间位置。

#### Step 3: 跨语言过渡区域

`zh→en` 和 `en→zh` 的间隙是因为 English token 在 pp tier 中可能被替换为 ARPABET phones，而替换过程中边界处理不精确。`_apply_en_phones` 操作的是 phones tier，后续 Phase 5 的 `build_pinyin_phones_tier` 在处理 English word 之前和之后的间隙时可能留下不连续。

#### Step 4: 标点周围

`punct→zh` 和 `zh→punct` 的间隙主要来自 `_inject_punctuation` 后 pp tier 重建时的边界对齐偏差。当标点被注入到 words tier 后，pp tier 中对应的 phone 可能没有完美同步。

### 影响函数

| 函数 | 文件:行号 | 角色 |
|------|----------|------|
| `build_pinyin_phones_tier` | postprocess_textgrids.py:473-480 | silence gap 处理 |
| `_apply_en_phones` | postprocess_textgrids.py:3885-3899 | English phone 替换边界 |
| `_inject_punctuation` (pp 重建) | postprocess_textgrids.py:2626-2650 | pp tier 的标点处理 |

### 修复方案

**修复 1 — words tier 间隙修复后自动传播到 pp tier**（依赖 Issue ⑤ 的修复）:

修复 words tier 间隙后，在 Phase 5 的 pinyin_phones 重建时 pp tier 会自动变连续。

**修复 2 — `build_pinyin_phones_tier` 显式填充 gaps**:

```python
# 在 build_pinyin_phones_tier 的 word loop 中，处理间隙时：
# 不仅插入 silence label，还从 phones tier 中查找匹配的 sil/sp
# 如果 phones tier 中没有，则用 silence_label 作为 fallback
```

**修复 3 — 确保 pp tier 边界和 words tier 完全对齐**:

在 Phase 5 末尾（line 4776-4827），pp tier 的 phone 边界 snap 逻辑（line 4799-4802）仅对非 English 词的 first phone 做了 snap。应该对 English 词也做同样处理：

```python
# line 4799 — 修改前:
if not is_en and new_pp_ivs[first].xmin > w_iv.xmin + 0.005:

# 修改后: 对所有词的首音素做 snap
if new_pp_ivs[first].xmin > w_iv.xmin + 0.005:
```

---

## <a name="summary"></a>汇总修复优先级

| 优先级 | Issue | 修复难度 | 影响范围 | 依赖 |
|--------|-------|---------|---------|------|
| **P0** | ③ 英文词音素不足 | 中 | 35% 英文词 | — |
| **P0** | ① 文本重叠 | 低 | 11% 文件 | — |
| **P1** | ⑤ Words/Hanzi 间隙 | 中 | 18-48% 文件 | — |
| **P1** | ② 英文 phone 边界偏移 | 中 | 80 处 | — |
| **P2** | ⑥ pinyin_phones 不连续 | 中 | 27% | ⑤ 修复后大部分自动解决 |
| **P2** | ④ 异常长词 | 低 | 1 文件 | — |

### 建议修复顺序

1. **先修 ③**（英文词音素不足）—— 影响面最大，且独立于其他修复
2. **再修 ①**（文本重叠）—— 低难度，消除 words tier 重叠后 ⑤ 和 ⑥ 也会受益
3. **修 ⑤**（间隙吸收）—— 提高 gap 吸收阈值
4. **修 ②**（phone 边界偏移）—— 确保 English phone 边界与 word 对齐
5. **修 ⑥**（pp 不连续）—— 大部分随 ①+⑤ 修复自动解决
6. **修 ④**（异常长词）—— 添加检测 + 保护

### 涉及文件

| 文件 | 修改范围 |
|------|---------|
| `scripts/postprocess_textgrids.py` | 主要修改: `_snap_to_ctc`, `_fix_overlapping_boundaries`, `build_pinyin_phones_tier`, `_apply_en_phones`, `process_one`, `detect_issues` |
| `scripts/align_english_mfa.py` | 轻微: 确保 `en_word_start`/`en_word_end` 精度 |
| `REGRESSION_ARCHIVE.md` | 追加新 Case 记录 |

### 验证方法

修复后对同样的 400 样本重跑 `postprocess` 步骤，对比：
1. 文本重叠数 → 应降至 < 1%
2. 英文词音素不足 → 应降至 < 5%
3. words/hanzi 间隙率 → Batch1 < 3%, Batch2 < 10%
4. pp 不连续率 → 应降至 < 5%
5. 无 > 3s 的短中文词
