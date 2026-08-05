# long_consonant_phone / long_vowel_phone 根因分析与修复方案

> **数据源**: `shayi_huali_new` filtered — 40 文件深度采样
> **分析日期**: 2026-08-05

---

## 一、问题分布

| 根因 | 发生率 | 典型表现 |
|------|--------|---------|
| RC1 — pp tier 结构重叠 | 32.5% 文件 | `，[4.000-4.750]` ↔ `z[4.610-4.659]` 重叠 140ms |
| RC2 — 声母占比异常 | 10.1% 音节 | `hao3` 270ms → `h`=190ms(70%), `ao3`=80ms(30%) |
| RC3 — 韵母被词边界拉长 | ~2% 音节 | `piao4` 1066ms → `p`=80ms, `iao4`=986ms(92%) |

---

## 二、各根因详细分析

### RC1 — pp tier 结构重叠 (13/40 files, 32.5%)

#### 现象

pp tier 中存在不同 label 的 Interval 重叠 10-155ms：

```
，[4.000-4.750]  ↔  z[4.610-4.659]   overlap=140ms  (punct↔content)
。[2.370-2.825]  ↔  …[2.670-2.825]   overlap=155ms  (punct↔punct)
x[4.840-4.920]  ↔  ，[4.840-9.180]   overlap=80ms   (content↔punct)
```

#### 根因链

1. `_inject_punctuation` 将 CTC 标点锚点注入 pp tier 时，使用标点的 CTC 时间戳
2. CTC 标点时间戳可能落在 content word 内部（因为 CTC 帧移 40ms 的精度限制）
3. pp tier 的标点 interval 和词的 phone interval 重叠
4. `_inject_punctuation` 的 pp 重建代码（~line 2626-2650）用 max/min 裁剪 phone 到 word 范围，但未处理 phone 和 punct 之间的重叠

关键代码 [postprocess_textgrids.py:2636-2642](scripts/postprocess_textgrids.py#L2636-L2642)：
```python
for p_iv in pp_tier.intervals:
    if p_iv.xmax > iv[0] and p_iv.xmin < iv[1] \
       and not is_silence(p_iv.text):
        word_phones.append(Interval(
            max(p_iv.xmin, iv[0]), min(p_iv.xmax, iv[1]),
            p_iv.text))
```

这里对 phone 做了 max/min clip 到 word 范围内，但**没有检查 phone 和 punct 是否在同一位置重叠**。当 punct interval 和 word interval 在时间上相邻或重叠时，同一段音频区间同时被 phone 和 punct 占据。

#### 修复方案

**策略**：标点保留 ≥60ms，重叠时标点优先占用边界的能量低谷区，phone 被裁剪到能量区。

```python
# 在 _inject_punctuation 的 pp 重建循环末尾（line 2650 前）添加：
# Resolve phone-punct overlaps in pp tier:
# - Punct keeps ≥ 60ms
# - When punct overlaps a content phone, clip the phone to the non-punct range
# - If phone falls entirely within punct, use proportional scaling 
#   to redistribute the word's phones within the available non-punct time
```

具体步骤：
1. 收集 pp tier 中所有 content phone 和 punct 的时间范围
2. 对每个重叠：punct 保留 ≥60ms，phone 被裁剪或用 proportional split 重建
3. 如果 word 内所有 phone 都被 punct 覆盖 → 用 proportional split 在剩余时间内重建 phone 序列

---

### RC2 — 声母占比异常 (142/1404 音节, 10.1%)

#### 现象

```
hao3  总长270ms → h=190ms(70%)  ao3=80ms(30%)    正常: h~80ms(30%)  ao3~190ms(70%)
de5   总长160ms → d=110ms(69%)  e5=50ms(31%)     正常: d~40ms(25%)  e5~120ms(75%)
shang4总长180ms → sh=130ms(72%) ang4=50ms(28%)   正常: sh~60ms(33%) ang4~120ms(67%)
```

声母占据词长的 60-75%，韵母被严重压缩。

#### 根因链

1. `build_pinyin_phones_tier` 在正常分支（`len(word_phones) >= 2`）使用 MFA 第一个 phone 的 `xmax` 作为声母→韵母分界点
2. MFA 在某些声学环境下（特别是擦音 h/sh/x/f/s + 元音的模糊过渡、前后音节韵母衔接处）把 phone boundary 放得过晚
3. 代码无条件信任 MFA 的 phone boundary：

```python
# postprocess_textgrids.py:607-610
new_intervals.append(Interval(w_iv.xmin, word_phones[0][1], dict_phones[0]))
```

4. 即使此前有 proportional split fallback（Case 26），它只在 `word_phones <= 1` 时触发。当 `word_phones >= 2` 时走正常分支，不做任何比例校验。

#### 修复方案

**策略**：不直接信任 MFA phone boundary，用 phonetically-motivated 比例上限做保护。如果 MFA boundary 给出的声母占比超过阈值，用 proportional split 替代。

声母时长上限（基于汉语语音学）：

| 声母类型 | 上限（占词长） | 典型音素 |
|---------|--------------|---------|
| 不送气塞音 | 35% | b, d, g |
| 送气塞音 | 40% | p, t, k |
| 擦音 | 50% | f, s, sh, x, h, r |
| 塞擦音 | 45% | z, c, zh, ch, j, q |
| 鼻音/边音 | 40% | m, n, l |

```python
# 在 build_pinyin_phones_tier 的正常分支中，添加比例保护：
_INIT_MAX_FRAC = {
    'b':0.35, 'd':0.35, 'g':0.35,          # 不送气塞音
    'p':0.40, 't':0.40, 'k':0.40,           # 送气塞音
    'f':0.50, 's':0.50, 'sh':0.50, 'x':0.50, 'h':0.50, 'r':0.50,  # 擦音
    'z':0.45, 'c':0.45, 'zh':0.45, 'ch':0.45, 'j':0.45, 'q':0.45, # 塞擦音
    'm':0.40, 'n':0.40, 'l':0.40,           # 鼻音/边音
}

if len(word_phones) >= 2:
    init_end = word_phones[0][1]
    init_frac = (init_end - w_iv.xmin) / word_dur
    max_frac = _INIT_MAX_FRAC.get(dict_phones[0], 0.50)
    if init_frac > max_frac:
        # MFA boundary gives too much to initial → proportional split
        split = w_iv.xmin + word_dur * (_INIT_FRAC.get(dict_phones[0], 0.35))
        split = max(split, w_iv.xmin + 0.030)
        split = min(split, w_iv.xmax - 0.030)
        new_intervals.append(Interval(w_iv.xmin, split, dict_phones[0]))
        final_label = " ".join(dict_phones[1:]) if len(dict_phones) > 2 else dict_phones[1]
        new_intervals.append(Interval(split, w_iv.xmax, final_label))
    else:
        # MFA boundary is reasonable — use it
        new_intervals.append(Interval(w_iv.xmin, init_end, dict_phones[0]))
        ...
```

**能量仲裁**（进阶）：如果 MFA boundary 和 proportional split 差异 > 50ms，检查 boundary 附近的能量。如果 proportional split 点处在能量低谷（低于 noise_floor × 3），则优先用 proportional split。

---

### RC3 — 韵母被词边界拉长 (24/1404 音节, ~2%)

#### 现象

```
piao4 总长1066ms → p=80ms  iao4=986ms(92%)   MFA原始: p~80ms iao4~290ms
ming2 总长700ms  → m=70ms  ing2=630ms(90%)   MFA原始: m~70ms ing2~200ms
```

声母保持 MFA 原始时长（正常），韵母被扩展填满词的拉伸边界。

#### 根因链

1. Postprocessing 把词边界拉长（CTC snap / silence absorption / NVV extension）
2. `build_pinyin_phones_tier` 用 MFA phone 边界放声母（正常），韵母的 start 来自 MFA phone
3. **Phase 5 的尾 phone 扩展逻辑**无条件把韵母 end 推到词尾：

```python
# postprocess_textgrids.py ~line 4951
if w_iv.xmax > new_pp_ivs[last].xmax + 0.005:
    extend_to = w_iv.xmax
    ...
    new_pp_ivs[last] = Interval(new_pp_ivs[last].xmin, extend_to, ...)
```

4. 没有检查扩展后的 phone 时长是否超过生理/声学合理范围

#### 修复方案

**策略**：尾 phone 扩展加绝对时长上限 + 能量验证。超出部分用 silence gap 表示。

```python
# Phase 5 尾 phone 扩展 — 添加保护：
_VOWEL_MAX_MS = 0.400   # 韵母绝对上限 400ms
_CONS_MAX_MS = 0.200    # 声母绝对上限 200ms
_SINGLE_PHONE_MAX_MS = 0.500  # 零声母单 phone 上限 500ms

if w_iv.xmax > new_pp_ivs[last].xmax + 0.005:
    last_dur = new_pp_ivs[last].xmax - new_pp_ivs[last].xmin
    is_vowel = not bool(re.match(r'^[bpmfdtnlgkhjqxrzcs]$|^[zcs]h$', new_pp_ivs[last].text))
    is_single = (first == last)
    
    if is_single:
        max_dur = _SINGLE_PHONE_MAX_MS
    elif is_vowel:
        max_dur = _VOWEL_MAX_MS
    else:
        max_dur = _CONS_MAX_MS
    
    extend_to = w_iv.xmax
    if last_dur + (extend_to - new_pp_ivs[last].xmax) > max_dur:
        # Would exceed max — cap the extension
        extend_to = new_pp_ivs[last].xmin + max_dur
    
    # Energy check: if extension region has no energy, don't extend
    if wav_audio is not None:
        ext_start = int(new_pp_ivs[last].xmax * sr)
        ext_end = int(extend_to * sr)
        if ext_end > ext_start:
            ext_energy = np.mean(np.abs(wav_audio[ext_start:ext_end]))
            if ext_energy < noise_floor * 2.0:
                extend_to = new_pp_ivs[last].xmax  # dead silence, don't extend
    
    new_pp_ivs[last] = Interval(new_pp_ivs[last].xmin, extend_to, new_pp_ivs[last].text)
```

---

## 三、约束条件满足分析

| 约束 | RC1 修复 | RC2 修复 | RC3 修复 |
|------|---------|---------|---------|
| 不丢失文本 | ✅ punct 保留 ≥60ms | ✅ 声母+韵母始终都创建 | ✅ silence gap 替代超限扩展 |
| CTC 锚点不受影响 | ✅ 仅修改 pp tier 内部结构 | ✅ 不改词边界 | ✅ 不改词边界 |
| 标点 ≥60ms | ✅ 显式保证 | N/A | N/A |
| 能量感知 | ✅ 用能量低谷区放置标点 | ✅ 进阶：能量仲裁 boundary | ✅ 扩展区无能量则不扩展 |
| 重叠时元素完整 | ✅ phone 被裁剪而非删除 | N/A | N/A |
| 比例缩放保证音节数 | ✅ 完全覆盖时用比例重建 | ✅ 比例 split 替代 MFA boundary | N/A |

---

## 四、实现优先级

| 优先级 | 修复 | 原因 |
|--------|------|------|
| **P0** | RC2 — 声母比例保护 | 影响 10.1% 音节，修复简单（只加比例检查），无副作用 |
| **P0** | RC3 — 尾 phone 时长上限 | 影响最严重 case（900ms+ 的韵母），加能量检查 |
| **P1** | RC1 — pp tier 重叠修复 | 影响 32.5% 文件但多数属 cosmetic；需要较复杂逻辑 |

---

## 五、涉及文件与函数

| 文件 | 函数 | 修改内容 |
|------|------|---------|
| `scripts/postprocess_textgrids.py` | `build_pinyin_phones_tier` | RC2: 正常分支加 `_INIT_MAX_FRAC` 比例保护 |
| `scripts/postprocess_textgrids.py` | `process_one` (Phase 5) | RC3: 尾 phone 扩展加 `max_dur` + 能量验证 |
| `scripts/postprocess_textgrids.py` | `_inject_punctuation` | RC1: pp 重建时解决 phone↔punct 重叠 |
