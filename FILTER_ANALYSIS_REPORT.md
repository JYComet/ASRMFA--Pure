# MFA Pipeline 过滤结果深度分析报告

> **数据来源**: `\\RS3621\Research_TTS\Data\Raw\ASR_MFA`  
> **处理配置**: `configs/batch_all.yaml`  
> **分析日期**: 2026-07-13  
> **总处理文件数**: 100,929 (161 个数据集)  
> **通过**: 98,072 (97.2%)  
> **被过滤**: 2,857 (2.8%)

---

## 一、总体过滤统计

### 1.1 过滤原因分布

| 过滤原因 | 数量 | 占比(总过滤) | 说明 |
|----------|------|-------------|------|
| `word_in_silence` | 2,686 | 94.0% | 词能量在静音水平 |
| `bgm_suspect` | 49 | 1.7% | 静音段检测到BGM/噪声 |
| `mid_sp` | 48 | 1.7% | 句中存在静音间隙 |
| `sp3` | 36 | 1.3% | 存在>1.5s长静音 |
| `unexpected_silence` | 31 | 1.1% | 词间静音无对应标点 |

### 1.2 对齐问题细节分布

| 问题类型 | 数量 | 说明 |
|----------|------|------|
| `word_in_silence` | 3,177 | 词能量低于噪声底×2.0 |
| `short_phone` | 1,229 | 音素时长<0.015s |

### 1.3 高过滤率数据集 (过滤率>5%)

| 数据集 | 总数 | 过滤数 | 过滤率 | 主要问题 |
|--------|------|--------|--------|----------|
| 鸣海派 | 1,963 | 431 | 21.9% | word_in_silence (直播BGM) |
| Xiaoyuan | 1,997 | 462 | 23.1% | word_in_silence (直播BGM) |
| 乃琳 | 57 | 10 | 17.5% | word_in_silence |
| 王宝煲 | 7 | 1 | 14.3% | word_in_silence |
| 冥冥 | 41 | 5 | 12.2% | word_in_silence |
| 嘉然 | 1,608 | 173 | 10.8% | word_in_silence |
| 奈奈莉娅 | 596 | 61 | 10.2% | word_in_silence |
| 花礼 | 1,614 | 157 | 9.7% | word_in_silence |
| 宣小纸不怕火 | 171 | 17 | 9.9% | word_in_silence |

---

## 二、详细过滤原因分析

### 2.1 `word_in_silence` — 词能量在静音水平 (最主要问题, 94%)

**检测逻辑**:
```
噪声底估计: 取全音频 RMS 帧的底部 15% 分位数作为噪声底 (noise_floor)
阈值: noise_floor × filter_word_energy_ratio (默认 2.0, 且至少 noise_floor × 10.0)
触发: word_rms < threshold 的词被标记
跳过: 英文/NVV词（MFA无法对其声学建模）、标点、相邻于英文/NVV的词
```

**能量值示例**:
```
gei3:  energy=0.001679, noise_floor=0.001122 → 低于阈值被过滤
de5:   energy=0.000948, noise_floor=0.000634 → 低于阈值
le5:   energy=0.003207, noise_floor=0.002103 → 边界偏低
shi4:  energy=5.2e-05  → 几乎完全在静音中
na4:   energy=1.6e-05  → 完全在静音区域
```

**根本原因分析**:

| 子原因 | 描述 | 占比估计 |
|--------|------|----------|
| **a. MFA边界漂移到静音区** | MFA在fine_tune阶段把短词(的/了/着/呢/吗/吧/啊/是/在/个/和/就/也/都/不/没)的边界推入相邻静音段，因为这些虚词声学特征弱，MFA的声学模型倾向于把静音分配给低置信度的词 | ~60% |
| **b. CTC锚点已在静音中** | NVASR的CTC解码在句首/句尾插入了过多的静音帧，导致该位置对应的第一个/最后一个词的锚点实际落在静音区域 | ~15% |
| **c. 直播BGM覆盖** | 游戏中角色语音+背景音乐同时存在，BGM使噪声底整体抬高，但短虚词的实际能量低于抬高后的阈值 | ~15% |
| **d. 不送气声母/轻声词** | 不送气声母(b/d/g/j/zh/z) + 轻声(de5/le5/zhe5)本身的声学能量极低，即使在正确边界内能量也接近噪声底 | ~5% |
| **e. 音频剪辑边缘** | 剪辑点附近的词只有部分音频，MFA被迫将边界延伸到静音 | ~3% |
| **f. 麦克风/录音问题** | 部分音频的信噪比本身就很低，导致短词能量难以与噪声区分 | ~2% |

**具体解决方案**:

1. **调整MFA fine_tune参数** — 减小 `fine_tune_boundary_tolerance` (当前 0.1s → 0.05s)，减少MFA将边界推入静音的自由度
2. **提高beam宽度** — 提高 `--beam` (当前20→30) 和 `--retry_beam` (当前80→120)，让MFA探索更多对齐路径
3. **调整boost_silence** — 降低 `--boost_silence` (当前1.0→0.8)，减少HMM中对静音的偏好
4. **提高filter_word_energy_ratio** — 当前阈值 2.0×noise_floor 很严格，可放宽到 3.0-5.0
5. **短词白名单** — 对高频短虚词(的/了/着/呢/吗/吧/啊/是/在/个/和/就/也/都/不/没)放宽word_in_silence检查
6. **分数据集调整阈值** — 对BGM严重的数据集(鸣海派/Xiaoyuan/嘉然/花礼)使用更宽松的阈值
7. **CTC锚点预清理** — 在adjust_ctc_boundaries阶段更激进地切除句首/句尾静音
8. **分频段能量检测** — 不在全频段做能量检测，而在语音主要频段(300-3400Hz)检测，排除BGM低频干扰

### 2.2 `bgm_suspect` — 静音段检测到BGM/背景噪声 (49个, 1.7%)

**检测逻辑**:
```
噪声底估计: silence-labeled帧的底部10%分位数
阈值: max(noise_floor × bgm_noise_floor_ratio, 0.005)
触发: 静音区间能量 > 阈值 且 > avg_speech_energy × bgm_speech_ratio
文件级决策: 任何静音区间触发 → 整个文件标记为bgm_suspect
```

**实际检测示例**:
```
noise_floor=1e-06 (极低)
avg_speech_energy=0.090516
suspect_interval: [8.86s-9.18s] energy=0.098541 → 静音段能量接近语音水平!
suspect_ratio: 0.066 (6.6%的静音段能量异常)
```

**根本原因分析**:

| 子原因 | 描述 |
|--------|------|
| **a. 背景音乐(BGM)** | 游戏/动画配音带有持续BGM，在语音停顿处BGM仍然存在，MFA正确标记了静音但该区域实际包含音乐能量 |
| **b. 环境噪声** | 录音环境持续噪声(风扇/空调/电流声) |
| **c. 混响尾音** | 大房间录音的混响尾音在静音标记段仍有余响 |
| **d. 多人对话** | 对话场景中，远处说话人的声音被标记为当前说话人的静音段 |
| **e. MFA silence标记不准确** | MFA将带能量的语音尾音误标记为sil |

**解决方案**:

1. **BGM分离预处理** — 使用Demucs/UVR等音源分离工具在MFA前剥离BGM轨道
2. **自适应BGM容忍** — 对已知BGM数据集降低bgm_speech_ratio阈值或完全跳过bgm检测
3. **分数据集标记** — 在config中对游戏/直播数据集设置 `detect_bgm: false`
4. **静音段频谱分析** — 不只检测能量，还检测频谱平坦度(spectral flatness)，真正BGM的频谱比语音噪声更不平均
5. **中位数噪声底** — 用中位数代替底部10%分位数计算噪声底，减少单个干净静音段的影响

### 2.3 `mid_sp` — 句中静音间隙 (48个, 1.7%)

**检测逻辑**:
```
检查words tier中第2个及之后的interval:
如果存在非空的silence label (<sp0>/<sp1>/<sp2>/<sp3>/sil/<eps>) → 标记为mid_sp
即：句中出现了静音标记，正常对齐后静音应该只在句首
```

**根本原因分析**:

| 子原因 | 描述 |
|--------|------|
| **a. 长停顿无标点** | 说话人在句中自然停顿但文本中没有标点符号，CTC注入的...或MFA插入的sil留在句中 |
| **b. NVV/英文词相邻** | MFA对英文词/NVV无法建模声学，在这些词附近产生spurious silence (spn→sil) |
| **c. CTC锚点错位** | CTC的一个词的锚点跨越了两个实际词，中间产生空隙 |
| **d. sp0合并失败** | handle_unexpected_silences合并<sp0>失败，残留在句中 |

**解决方案**:

1. **改进silence合并** — 扩大merge_max_sil_sec (0.2s→0.3s) 或降低merge_energy_threshold
2. **sp0→前词合并** — 对句中残留的所有<sp0>无条件合并到前一个词
3. **动态静音标记** — 句中的<sp1>如果<0.3s且无标点→直接合并；如果>0.3s→标记为[sp]并保留
4. **英文/NVV相邻静音处理** — 英文/NVV词后的静音自动合并
5. **重新注入标点** — 句中长停顿(>0.5s)自动注入逗号，使文本与音频一致

### 2.4 `sp3` — 长静音>1.5s (36个, 1.3%)

**检测逻辑**:
```
words tier中任何interval的text == "<sp3>" → 标记
<sp3> = 时长>1.5s的静音
```

**根本原因分析**:

| 子原因 | 描述 |
|--------|------|
| **a. 句首/句尾未裁剪** | 音频开头或结尾有长段静音未被trim掉 |
| **b. 句子间长停顿** | 多句音频中两句之间的自然长停顿被保留 |
| **c. CTC停顿检测过度** | NVASR blank-run检测将语音中的自然停顿标记为长静音 |
| **d. 音频质量问题** | 录音中有片段缺失/静音段 |
| **e. 音频trim步骤未执行** | ctc_ready模式下跳过了trim步骤，直接从已有音频处理 |

**解决方案**:

1. **预处理静音裁剪** — 在ctc_ready模式下也增加句首/句尾silence trim步骤
2. **调整CTC pause_frames** — 增大PAUSE_FRAMES阈值(当前8帧≈480ms→12帧≈720ms)，减少虚假停顿检测
3. **sp3→标点转换** — 句中sp3转换为句号或段落分隔符，句首/句尾sp3直接裁剪
4. **后处理裁剪** — 在postprocess中自动切除首尾的<sp3>区间

### 2.5 `unexpected_silence` — 词间静音无标点 (31个, 1.1%)

**检测逻辑**:
```
对比pinyin text中的标点位置与words tier中的实际silence位置:
- 有silence无标点 → 短silence(<sp0>)合并到前词
- 有silence无标点 → 长silence(<sp1>/<sp2>/<sp3>)标记为unexpected_silence
- 跳过英文/NVV相邻的silence (MFA产生的artifact)
```

**根本原因分析**:

| 子原因 | 描述 |
|--------|------|
| **a. 说话人自然停顿** | 口语中自然有停顿但文本没有标记(直播/对话场景常见) |
| **b. CTC标点丢失** | NVASR在解码时漏掉了部分标点，导致某些位置有停顿无标点 |
| **c. MFA silence注入** | MFA的对齐过程中在低置信度区域插入silence |
| **d. 多个sp1-3残留** | 一个文件中多个词间silence都没被合并/转换 |

**解决方案**:

1. **自动标点注入** — 对>0.4s的unexpected silence自动插入逗号，>0.8s自动插入句号
2. **阈值调高** — 提高unexpected_silence的最小检测阈值到0.5s，忽略较短的停顿
3. **NVASR标点召回** — 检查ctc_prealign的blank-run标点注入逻辑，确保长停顿正确标记
4. **合并策略优化** — <sp2>也尝试合并(当前只合并<sp0>)，前提是前后词的音素能够连读

### 2.6 `short_phone` — 音素过短 (1,229个实例)

**检测逻辑**:
```
pinyin_phones tier中:
  - 非silence/spn音素时长 < filter_short_phone_sec (默认 0.015s)
  - 跳过英文/NVV区间(这些MFA无法声学建模)
```

**常见短音素**: `w`, `d`, `ə˥˩`, `ə˧˥`, `e˨˩˦`, `uo1` 等

**根本原因分析**:

| 子原因 | 描述 |
|--------|------|
| **a. 不送气声母截断** | b/d/g/j/zh/z等不送气声母(IPA: p/t/k/tɕ/ʈʂ/ts)声学能量弱，MFA给它们分配极短时长 |
| **b. 半元音过短** | y/w等半元音(glide)被MFA压缩到极短(<10ms) |
| **c. 词边界处音素碎裂** | 两个相邻词的边界音素被CTC锚点强制切开，碎片化 |
| **d. 送气段被吸收** | 送气声母(p/t/k/q/ch/c)的送气段被MFA归入前一个silence |
| **e. MFA fine_tune副作用** | fine_tune把音素边界向CTC锚点压缩 |

**解决方案**:

1. **调高short_phone阈值** — 从0.015s提高到0.02s，减少低质量过滤
2. **不送气声母白名单** — b/d/g/j/zh/z对应的IPA音素(p/t/k/tɕ/ʈʂ/ts)放宽short_phone检查到0.008s
3. **半元音特殊处理** — y/w/j/ɥ等过渡音素放宽阈值
4. **减少fine_tune自由度** — 降低fine_tune_boundary_tolerance，避免音素被过度压缩
5. **音素级后处理合并** — 对过短的相邻音素进行声学合理的合并

### 2.7 `word_without_phone` — 词没有对应的音素

**检测逻辑**:
```
某个词区间在phones/pinyin_phones tier中找不到任何非静音音素 → 标记
```

**根本原因**:

| 子原因 | 描述 |
|--------|------|
| **a. MFA对齐完全失败** | 该词在MFA词典中找不到或声学得分过低 |
| **b. 英文/NVV词的词典缺失** | 英文词不在MFA的IPA词典中 |
| **c. 音频中该词缺失** | CTC标注了但实际音频中没有这个词 |

**解决方案**:
1. 扩充MFA词典覆盖所有英文/NVV token
2. 对词典外单词使用G2P生成发音
3. 降低acoustic_scale让MFA更宽松匹配

### 2.8 `low_phone_coverage` — 音素覆盖率低

**检测逻辑**:
```
phone_coverage = sum(phone在word区间内的时长) / word时长
如果 < filter_min_phone_coverage (默认 0.35) → 标记
```

**根本原因**:

| 子原因 | 描述 |
|--------|------|
| **a. MFA将部分音素推到词边界外** | fine_tune导致音素边界与词边界不重合 |
| **b. 词间有大段静音** | silence占据了词区间的大部分 |
| **c. 长词被压缩** | 长词的phone被过度压缩到词区间的一小部分 |

**解决方案**:
1. 调整fine_tune参数，确保音素边界与词边界一致
2. 提高filter_min_phone_coverage 到 0.5 以捕获更多问题
3. 对压缩的长词进行专门检测

### 2.9 `large_edge_gap` — 词-音素边界间隙过大

**检测逻辑**:
```
word_start - first_phone_start > filter_edge_gap_sec (默认 0.25s)
或 last_phone_end - word_end > filter_edge_gap_sec → 标记
```

**根本原因**:
- CTC锚点与MFA音素边界的时间差过大
- MFA在词边界处插入了不合理的silence

**解决方案**:
1. 减小edge_gap阈值到0.15s以提高敏感度
2. 在后处理中将过大gap分配给相邻silence

### 2.10 `word_too_short` — 词时长过短

**检测逻辑**:
```
word duration < filter_min_word_dur_sec (默认 0.02s)
```

**根本原因**: MFA将某些短虚词压缩到极短

**解决方案**:
1. 将过短词合并到相邻词
2. 或者使用CTC锚点强制扩展边界

### 2.11 `long_word` — 词时长过长

**检测逻辑**:
```
word duration > filter_long_word_sec (默认 1.0s)
```

**根本原因**: 慢语速、拖长音、或MFA将多个词合并

**解决方案**: 调高阈值或标记后人工审核

### 2.12 预处理阶段可能的问题 (当前未在筛查报告中)

这些是预处理阶段可能产生的问题，当前配置大部分已处理好：

| 问题 | 阶段 | 影响 |
|------|------|------|
| **CTC模型失败** | ctc_prealign | NVASR GPU OOM、模型加载失败、超时 |
| **音频重采样失败** | resample | 损坏或格式不兼容的WAV |
| **MFA词典OOV** | validate | 英文/NVV不在MFA词典中(已通过自指涉解决) |
| **MFA对齐超时** | align | 长音频(>30s)可能超时 |
| **CTC文件不完整** | link | 缺少6个CTC输出文件中的任意1个→跳过该stem |
| **WAV文件缺失** | link | audio_root中没有对应stem的wav文件 |
| **标点归一化失败** | normalize_punct | 中文标点映射表缺失 |
| **数字归一化失败** | normalize | cn2an转换异常 |

---

## 三、解决方案优先级矩阵

### Tier 1: 立即实施 (高影响、低成本)

| # | 解决方案 | 目标问题 | 预期改善 | 实施方式 |
|---|----------|----------|----------|----------|
| 1 | **放宽word_in_silence阈值** | word_in_silence (94%) | 减少50-70%假阳性 | `filter_word_energy_ratio: 2.0→4.0` |
| 2 | **短虚词白名单** | word_in_silence (短词) | 减少30%假阳性 | 代码修改: 的/了/着/呢/吗/吧 等词跳过word_in_silence |
| 3 | **分数据集配置** | BGM数据集高过滤率 | 鸣海派等从23%降到5% | 在batch_all.yaml中增加per-dataset override |
| 4 | **句首句尾sp3裁剪** | sp3 (36个) | 消除大部分sp3 | postprocess中自动切除首尾长静音 |

### Tier 2: 短期实施 (高影响、中成本)

| # | 解决方案 | 目标问题 | 预期改善 |
|---|----------|----------|----------|
| 5 | **MFA参数调优** | short_phone + word_in_silence | beam 20→30, retry_beam 80→120, boost_silence 1.0→0.8 |
| 6 | **分频段能量检测** | BGM误检测 | 语音频段(300-3400Hz)带通滤波后检测，排除BGM低频 |
| 7 | **CTC边界预裁剪** | word_in_silence (边缘) | adjust_ctc中更激进的句首句尾silence切除 |
| 8 | **自动标点注入** | unexpected_silence + mid_sp | >0.4s停顿→逗号, >0.8s→句号 |

### Tier 3: 中期实施 (中影响、中高成本)

| # | 解决方案 | 目标问题 | 预期改善 |
|---|----------|----------|----------|
| 9 | **BGM分离预处理** | bgm_suspect + word_in_silence | Demucs/UVR剥离BGM轨道后对齐 |
| 10 | **不送气声母特殊处理** | short_phone | b/d/g/j/zh/z放宽到0.008s |
| 11 | **fine_tune参数调优** | short_phone + word_in_silence | boundary_tolerance 0.1→0.05 |
| 12 | **MFA词典扩充** | word_without_phone | 为所有英文/NVV token生成IPA发音 |
| 13 | **音频质量预检** | 整体质量 | 预处理时检测信噪比，低质量音频标记 |

### Tier 4: 长期实施 (架构改进)

| # | 解决方案 | 目标问题 | 预期改善 |
|---|----------|----------|----------|
| 14 | **两级对齐策略** | 所有对齐问题 | 先用宽松参数快速对齐→问题文件用严格参数重对齐 |
| 15 | **音素级后处理合并** | short_phone + word_in_silence | 声学合理的相邻短音素自动合并 |
| 16 | **人工审核队列** | 疑难文件 | 自动生成Praat可视化，按优先级排队人工审核 |
| 17 | **自适应阈值系统** | 所有过滤 | 按数据集统计自动调整各过滤阈值 |

---

## 四、建议的配置修改

### 4.1 全局后处理参数调整 (configs/batch_all.yaml)

```yaml
postprocess:
  filter_suspicious: true
  enable_text_correction: true
  handle_unexpected_sil: true

  # ── 放宽word_in_silence ──
  filter_word_energy_ratio: 4.0    # 原 2.0 → 4.0 (减少假阳性)
  enable_word_in_silence_filter: true

  # ── 放宽short_phone ──
  filter_short_phone_sec: 0.02     # 原 0.015 → 0.02

  # ── 放宽phone_coverage ──
  filter_min_phone_coverage: 0.25  # 原 0.35 → 0.25

  # ── sp3自动裁剪 ──
  auto_trim_sp3: true              # 新参数

  # ── 自动标点注入 ──
  auto_inject_punctuation: true    # 新参数: >0.4s停顿→逗号
  auto_punct_threshold: 0.4
```

### 4.2 MFA参数调优

```yaml
mfa:
  num_jobs: 32
  single_speaker: true
  beam: 30                          # 原 20 → 30
  retry_beam: 120                   # 原 80 → 120
  boost_silence: 0.8                # 原 1.0 → 0.8 (减少静音偏好)
  fine_tune: true
  fine_tune_boundary_tolerance: 0.05  # 原 0.1 → 0.05 (减少边界漂移)
```

### 4.3 分数据集配置示例

```yaml
# 在 batch_all.yaml 中增加 dataset_overrides
dataset_overrides:
  # BGM严重的数据集 — 跳过BGM检测
  "鸣海派":
    postprocess:
      detect_bgm: false
      filter_word_energy_ratio: 6.0
  "Xiaoyuan":
    postprocess:
      detect_bgm: false
      filter_word_energy_ratio: 5.0
  "嘉然":
    postprocess:
      detect_bgm: false
      filter_word_energy_ratio: 5.0
  "花礼":
    postprocess:
      detect_bgm: false
      filter_word_energy_ratio: 5.0
  "奈奈莉娅":
    postprocess:
      detect_bgm: false
      filter_word_energy_ratio: 5.0
```

---

## 五、当前过滤结果健康度评估

**总体评价**: 过滤系统运行良好。97.2%的文件通过QC，说明MFA+CTC锚点的流水线整体可靠。

**需要关注的问题**:
1. **word_in_silence 假阳性率可能偏高** — 2,686个过滤中，估计30-50%可能是假阳性（短虚词在正确边界内能量就是偏低）
2. **BGM数据集的过滤率偏高** — 鸣海派(22%)、Xiaoyuan(23%)主要是直播BGM造成的系统性过滤
3. **short_phone部分为正常现象** — 不送气声母的时长通常就在10-15ms左右

**建议的行动计划**:
1. 先实施Tier 1的4个低成本改进
2. 重新运行后观察过滤率变化
3. 对剩余的过滤文件抽样人工审核(每类抽10个)
4. 根据人工审核结果调整Tier 2-4的参数
