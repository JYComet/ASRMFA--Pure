# Chinese MFA Forced Alignment Pipeline

基于 Montreal Forced Aligner (MFA) + NVASR CTC 强制对齐的中文音频标注管线，输入 wav + 中文文本，输出 5 层 Praat TextGrid。

## 前提条件

- Conda (Miniconda 或 Anaconda)
- GPU with CUDA (可选，CPU 也可运行但较慢)
- NVASR 模型文件 (~2.8 GB，需单独下载放入 `models/Multilingual-NVASR/`)

## 移植到新机器

整个 `chinese_mfa_pipeline/` 目录是**完全可移植的**——所有配置使用相对路径。新机器上只需：

1. **安装 Python 环境** — 运行一键脚本:
   ```bash
   setup_env.bat       # Windows
   bash setup.sh       # Linux/macOS
   ```
   这会创建 conda 环境 `mfa_chinese`、转换词典、下载 MFA 模型。

   如需 NVASR (CTC 预对齐)，还需要一个带 `funasr` + `torch` 的 Python 环境（可以是 base conda）。

2. **放入 NVASR 模型** — 将 Multilingual-NVASR 模型文件放入:
   ```
   models/Multilingual-NVASR/
   ├── model.pt
   ├── am.mvn
   ├── config.yaml
   ├── configuration.json
   └── paralingustic_tokenizer.model
   ```

3. **放入数据** — 在 `data_dir/` 下按子目录放 wav + txt 文件，或通过 `--data-dir` 指定外部路径。

## 目录结构

```
chinese_mfa_pipeline/
├── config.yaml                    # 全局默认配置
├── configs/                       # 任务配置
│   ├── xiaoyuan5.yaml
│   ├── xiaoyuan100.yaml
│   ├── bushi.yaml
│   └── batch_all.yaml
├── environment.yml                # conda 环境定义 (mfa_chinese)
├── requirements.txt               # pip 依赖
├── setup.sh / setup_env.bat       # 一键安装
├── scripts/
│   ├── run_pipeline.py            # 主管线 (8 步编排)
│   ├── streaming_pipeline.py      # 批量流式管线 (多数据集并行)
│   ├── pipeline_utils.py          # 共享工具 (路径翻译、文件发现、MFA 环境)
│   ├── trim_silence_batch.py      # 静音裁剪 + 首尾补全 (step 1)
│   ├── ctc_prealign.py            # NVASR CTC 强制对齐 → MFA 锚点 (step 3)
│   ├── adjust_ctc_boundaries.py   # 能量分析边界修正 (step 5)
│   ├── normalize_english_tokens.py # 英文 token 规范化 (step 4)
│   ├── postprocess_textgrids.py   # 后处理: 5 层构建 + 质检 + BGM (step 8)
│   ├── audio_energy.py            # 向量化音频能量分析
│   ├── audio_utils.py             # 音频重采样工具
│   ├── convert_dict_to_ipa.py     # 词典: 拼音 → IPA (setup 用)
│   ├── annotate_nvv.py            # NVV 副语言标注 (独立工具)
│   ├── view_in_praat.py           # 匹配 TextGrid + 音频用 Praat 打开
│   ├── verify_mapping.py          # IPA↔拼音 映射验证
│   ├── finalize_textgrids.py      # TextGrid 最终清理 (NVV 括号、<sp1> 规范化)
│   └── add_english_to_dict.py     # 英文 token 词典维护
├── dict/
│   ├── fullpinyin_enword.dict     # 拼音词典 (pypinyin 生成)
│   └── mfa_ipa.dict              # IPA 词典 (MFA 用)
├── models/
│   ├── mfa/                       # MFA 声学模型 + G2P 模型
│   └── Multilingual-NVASR/        # NVASR CTC 对齐模型
├── data_dir/                      # 输入数据 (wav + txt, 按子目录)
└── workspace/                     # 管线输出 (自动创建)
```

## 快速开始

### 1. 安装环境 (仅首次)

```bash
setup_env.bat       # Windows
bash setup.sh       # Linux / macOS
```

### 2. 放入数据

```
data_dir/{task_name}/
├── audio_001.wav          # 16kHz+ 单声道 WAV
├── audio_001.txt          # 同名中文文本 (UTF-8), 可选
├── audio_002.wav
├── audio_002.txt
└── ...
```

无参考文本时管线会使用 NVASR ASR 输出作为文本，同样可用。

文本文件若带引擎后缀（如 `audio_001_qwen3-api.txt`），管线自动匹配同名 wav。

### 3. 创建任务配置（2 行即可）

每个任务只需一个 YAML 文件，指定 **输出目录名** 和 **输入数据路径**：

```yaml
# configs/my_task.yaml
workspace: my_task          # 输出文件夹名
data_dir: data_dir/my_task  # 输入数据路径 (相对项目根, 或绝对路径)
```

所有其他参数都有内置默认值，无需重复编写。查看 [`config.yaml`](config.yaml) 了解全部可选字段。

### 4. 运行

```bash
# 全流程
python scripts/run_pipeline.py --config configs/my_task.yaml

# 跳过前几步, 从 prealign 开始 (数据已预处理好时)
python scripts/run_pipeline.py --config configs/my_task.yaml --skip-to prealign --overwrite

# 只跑单步
python scripts/run_pipeline.py --config configs/my_task.yaml --step postprocess

# 覆盖已有输出
python scripts/run_pipeline.py --config configs/my_task.yaml --overwrite
```

### 让 AI 帮你创建配置

在 Claude Code 中直接说：

> "处理 `data_dir/xxx` 下的音频，编写对应的配置文件"

AI 会自动：
1. 检查数据目录（文件数量、有无参考文本）
2. 创建最小的 2 行配置文件
3. 运行管线，报告结果

如果想调整参数，如：

> "处理 `data_dir/xxx`，静音裁剪阈值调到 0.02，MFA 用 4 线程"

AI 只会在配置中覆写这两个字段，其余沿用默认值。

## 输入数据格式

```
data_dir/{task_name}/
├── audio_001.wav          # 16kHz+ 单声道 WAV
├── audio_001.txt          # 同名中文文本 (UTF-8)
├── audio_002.wav
├── audio_002.txt
└── ...
```

文本文件是纯中文，可包含标点符号（，。！？…）。支持 NVV 标签格式 `[Breathing]` `[Laughter]` 等，管线会自动转换为 MFA 大写 token。

若文本文件带有引擎后缀（如 `audio_001_qwen3-api.txt`），管线会自动匹配到同名 wav。在配置中设置 `txt_suffix: qwen3-api` 可只匹配特定后缀。

## 管线步骤

| 步骤 | 名称 | 脚本 | 说明 |
|------|------|------|------|
| 1 | `trim` | `trim_silence_batch.py` | 内部静音裁剪 + 首尾补全到 0.5s |
| 2 | `resample` | (内联) | 降采样到 16kHz (MFA 要求) |
| 3 | `prealign` | `ctc_prealign.py` | NVASR CTC 强制对齐 → MFA 锚点 TextGrid |
| 4 | `normalize` | (内联) + `normalize_english_tokens.py` | cn2an 阿拉伯数字→中文数字 + 英文 token 规范化 |
| 5 | `adjust` | `adjust_ctc_boundaries.py` | 能量分析修正 CTC 锚点边界 |
| 6 | `validate` | MFA CLI | MFA 语料验证 |
| 7 | `align` | MFA CLI | MFA 声学模型对齐 (CTC 锚点 + NVASR 语料) |
| 8 | `postprocess` | `postprocess_textgrids.py` | 5 层 TextGrid 构建、标点注入、质检、BGM 检测 |

使用 `--list-steps` 查看所有步骤，`--skip-{step}` 跳过某步，`--skip-to {step}` 从某步开始。

## 输出结构

### 中间产物 (workspace/)

```
workspace/
├── audio/                  # 静音裁剪后的 WAV (原始采样率)
├── ctc_pretg/              # CTC 强制对齐输出
│   ├── *.TextGrid          # MFA 锚点 (words tier)
│   ├── *.lab               # MFA 语料文本 (拼音+NVV, 与 TextGrid 同源)
│   ├── *_tokens.jsonl      # 逐词 CTC 时间戳
│   ├── *_punct.json        # 标点 CTC 锚点
│   ├── *_text_cn.txt       # ASR 中文文本
│   ├── manifest.json       # 文件索引
│   └── summary.txt         # 统计报告
├── ctc_pretg_adj/          # 能量修正后的 CTC 锚点
├── aligned/                # MFA 对齐原始 TextGrid (words + phones)
├── output/                 # 最终 TextGrid (通过质检)
│   ├── *.TextGrid          # 5 层 TextGrid
│   ├── tone_mapping.json   # IPA↔拼音 声调映射表
│   └── postprocess_report.jsonl  # 处理报告
├── filtered/               # 未通过质检的 TextGrid
└── temp/                   # MFA 临时文件 + 16kHz 音频
```

### 最终 TextGrid (5 层)

| 层 | 内容 | 示例 |
|----|------|------|
| `raw_text` | 修正后的中文句子 | `<sp1>今天天气不错，我们出去玩` |
| `pinyin` | 拼音 + 标点 | `jin1 tian1 tian1 qi4 bu2 cuo4 ， wo3 men5 chu1 qu4 wan2` |
| `hanzi` | 每词一个汉字 / 静音标记 | `今` `天` `天` `气` `不` `错` `，` `我` `们` `出` `去` `玩` |
| `words` | MFA 对齐音节 + 标点 + 静音 + NVV | `jin1` `tian1` `tian1` `qi4` `bu2` `cuo4` `，` `<sp0>` `wo3` `men5` `chu1` `qu4` `wan2` |
| `pinyin_phones` | IPA→拼音音素 1:1 映射 | `j` `in1` `t` `ian1` `t` `ian1` `q` `i4` ... |

静音分级：`<sp0>` < 0.2s, `<sp1>` < 0.5s, `<sp2>` < 1.5s, `<sp3>` >= 1.5s

## 核心算法

### 1. NVASR CTC 强制对齐 (`ctc_prealign.py`)

用 NVASR (SenseVoice-Small 微调) 的 CTC logits 做强制对齐，而非自由解码:

- **参考文本优先**: 有参考文本时用 ground truth 中文做对齐，否则回退到 ASR 文本
- **Blank-frame NVV bias**: 对 CTC blank 帧的 NVV token logits 加偏置 (默认 4.0)，提升呼吸/笑声等检测
- **长停顿检测**: 连续 >=8 帧 blank (~480ms) → 注入省略号标记
- **Query frame 补偿**: 编码器前 4 帧为 lang/emo/textnorm query embedding，对齐时从 logits 中移除

### 2. CTC 边界能量修正 (`adjust_ctc_boundaries.py`)

在 MFA 对齐前用音频能量分析修正 CTC 锚点:

- **句首/标点后词首**: 检测静音残留，推后 start (节能 rise detection)
- **句尾/标点前词尾**: 检测语音截止，延长 end (fall detection)；或缩短多余的静音尾
- **标点同步**: 修正词边界时同步调整标点位置
- **NVV 保护**: 不对 NVV token 做边界修正

### 3. MFA/CTC 混合边界 (`_snap_to_ctc`)

MFA 对齐后，将 MFA 词边界与 CTC 锚点对比，混合取优:

```
对每个词:
  |MFA - CTC| <= 0.3s        → 信任 MFA (MFA 音素级精调更准)
  |MFA - CTC| > 0.3s         → snap 到 CTC (MFA 可能错位)

  例外:
  - NVV token → 始终用 CTC (MFA 无 NVV 声学模型)
  - MFA 词长 < 60ms 且 CTC > 150ms → 信任 CTC (短词保护, 如 yi4)
  - MFA 被信任但差异 > 0.15s → 中间点折中
  - word_start = max(word_start, prev_end) → 防词间重叠
```

### 4. 标点注入 (`_inject_punctuation`)

标点没有声学实现 (MFA 会转为 `<eps>`)，但有 CTC 时间戳。注入过程以**词优先**为原则:

- 词-标点重叠 → 裁剪标点，保护词 (不破坏音素完整性)
- 微小间隙 → 合并到标点/NVV (<=500ms)
- 残余间隙 → 优先分配给后一词的 start
- NVV 前方间隙 <=200ms → 吸收进 NVV (NVV 天然含周围静音，但有标点时跳过)
- 标点右边界 → 延伸到下个词 start

### 5. 标点-静音交叉校验 (`build_corrected_text`)

对比拼音文本的标点与实际 words tier 的静音间隙:

- 有标点但无静音 → 从文本删除该标点
- 无标点但有静音 → 插入 `[sp]` 标记

### 6. NVV + 省略号能量合并 (`_merge_nvv_ellipsis`)

NVV (如 LAUGHTER, BREATHING) 后的省略号 `...` 如果包含可听能量 (>=30% 帧 RMS > 噪声底x3)，则合并到 NVV，仅留 60ms 作为标点标记。

## NVV 副语言标签

管线支持 30 类 NVV (Non-Verbal Vocalization) 标签，由 NVASR 模型从音频中自动检测:

| 类别 | 标签 | 类别 | 标签 |
|------|------|------|------|
| 呼吸 | BREATHING | 笑声 | LAUGHTER |
| 咳嗽 | COUGH | 打嗝 | BURP |
| 哭泣 | CRYING | 呻吟 | GROAN |
| 嘶声 | HISS | 哼声 | HUM |
| 嘘声 | SHH | 叹气 | SIGH |
| 喷嚏 | SNEEZE | 抽鼻 | SNIFF |
| 打鼾 | SNORE | 啧啧 | TSK |
| 呃/嗯 | UHM | 口哨 | WHISTLE |
| 哈欠 | YAWN | | |
| 疑问-咦 | QUESTION-YI | 疑问-嗯 | QUESTION-EN |
| 疑问-哦 | QUESTION-OH | 疑问-啊 | QUESTION-AH |
| 疑问-诶 | QUESTION-EI | 疑问-哈 | QUESTION-HUH |
| 惊讶-哦 | SURPRISE-OH | 惊讶-啊 | SURPRISE-AH |
| 惊讶-哇 | SURPRISE-WA | 惊讶-哟 | SURPRISE-YO |
| 确认-嗯 | CONFIRMATION-EN | 不满-哼 | DISSATISFACTION-HNN |

这些标签在 MFA 词典中作为自指词条 (self-referential，如 `BREATHING: B R EA TH I NG`)，不在 MFA 声学模型中，因此 `_snap_to_ctc` 会直接用 CTC 锚点时间戳。

## 质检与过滤

后处理阶段对每个 TextGrid 做自动质检，不通过的放入 `filtered/`:

| 规则 | 默认阈值 | 说明 |
|------|----------|------|
| `short_phone` | < 0.005s | 音素过短 (对齐失败) |
| `long_word` | > 1.5s | 音节过长 (可能漏标点) |
| `word_too_short` | < 0.02s | 词过短 (错位) |
| `word_in_silence` | 能量 < 噪声底 x 2.0 | 词标在静音区域 |
| `low_phone_coverage` | < 25% | 词内音素覆盖不足 |
| `large_edge_gap` | > 0.35s | 词-音素边界间隙过大 |
| `short_word_between_silences` | < 0.12s + 两侧 > 0.4s | 孤立短词 |
| `bgm_suspect` | 静音段能量过高 | 背景音乐/噪声残留 |
| `unexpected_silence` | >= 0.2s 无标点停顿 | 意外长停顿 |
| `sp3` | >= 1.5s 静音 | 过长停顿 |
| `mid_sp` | 音频中间有静音标记 | 对齐不完整 |

## 配置完整参考

```yaml
# ── 路径 (相对项目根, 也支持绝对路径) ──
workspace: workspace          # 输出工作区
data_dir: data_dir            # 输入数据根目录
txt_suffix: ""                # 只匹配特定后缀的 txt (如 qwen3-api)

# ── 模型 & 词典 (相对项目根) ──
models_dir: models/mfa
acoustic_model: mandarin_mfa
mfa_dict: dict/mfa_ipa.dict
pinyin_dict: dict/fullpinyin_enword.dict

# ── Python 环境 (空 = 自动检测) ──
python_path: ""               # MFA Python。自动搜索 conda env: mfa_mandarin / mfa_chinese / mfa

# ── 输出子目录 (相对 workspace) ──
audio_dir: audio
pinyin_dir: pinyin
aligned_dir: aligned
output_dir: output
filtered_dir: filtered
validate_dir: validate
temp_dir: temp
ctc_pretg: ctc_pretg
ctc_pretg_adj: ctc_pretg_adj

# ── Step 1: 静音裁剪 ──
trim:
  max_silence_sec: 1.0          # 内部静音最长保留
  sil_vol_threshold: 0.005      # RMS 静音阈值
  sil_len_threshold: 0.08       # 最小静音段长度 (s)
  normalize_edges: true         # 规范化首尾静音
  target_edge_silence_sec: 0.5  # 首尾目标静音长度
  edge_silence_threshold: 0.001 # 首尾检测阈值
  edge_frame_length: 1024       # 首尾检测帧长
  target_sr: null               # 输出采样率 (null = 不变)
  workers: 8                    # 并行线程

# ── Step 3: CTC 预对齐 ──
ctc_prealign:
  enabled: true
  model_path: "models/Multilingual-NVASR"
  device: cuda:0
  python: ""                    # NVASR Python。空 = 当前 Python
  limit: 0                      # 0 = 全部
  timeout: 3600

# ── Step 5: CTC 边界修正 ──
ctc_adjust:
  enabled: true
  limit: 0

# ── Step 6-7: MFA ──
mfa:
  num_jobs: 8                   # 并行数
  single_speaker: true          # 单说话人模式
  output_format: long_textgrid
  clean: true                   # 清理临时文件
  no_tokenization: true         # 不使用 MFA tokenizer (用词典直接匹配)

# ── Step 8: 后处理 ──
postprocess:
  merge_silence: true
  min_sil_merge_sec: 0.2        # 短于此时长的静音可能被合并
  fix_short_word: true
  short_word_max_sec: 0.25      # 短词检测阈值
  flank_silence_sec: 0.4        # 短词两侧所需静音
  short_word_search_window: 0.5 # 短词后语音搜索窗口
  detect_bgm: true
  bgm_noise_floor_ratio: 2.0    # 静音能量 > 噪声底 x N → 可疑
  bgm_min_sil_dur: 0.3          # 最小静音段检查时长
  bgm_speech_ratio: 1.0         # 静音能量 > 语音 x N → 可疑
  bgm_min_energy: 0.01          # 触发绝对 RMS 阈值
  filter_suspicious: true
  filter_short_phone_sec: 0.005
  filter_long_word_sec: 1.5
  filter_min_word_sec: 0.15
  filter_min_word_dur_sec: 0.02
  filter_word_energy_ratio: 2.0
  filter_min_phone_coverage: 0.25
  filter_edge_gap_sec: 0.35
  filter_flank_silence_sec: 0.4
  filter_long_consonant_sec: 999.0   # 999 = 禁用
  filter_long_vowel_sec: 999.0       # 999 = 禁用
  enable_text_correction: true       # 标点↔静音交叉校验
  handle_unexpected_sil: false       # 合并无标点短停顿
```

## CLI 参考

```
python scripts/run_pipeline.py [OPTIONS]

Options:
  --config PATH      配置文件路径 (默认: config.yaml)
  --data-dir PATH    覆盖输入目录
  --output-dir PATH  覆盖输出目录
  --python PATH      覆盖 MFA Python 路径
  --step NAME        只运行指定步骤
  --skip-to NAME     从指定步骤开始
  --skip-{step}      跳过指定步骤
  --overwrite        覆盖已有输出
  --force            遇错继续执行
  --list-steps       列出所有步骤
```

### 常用模式

```bash
# 从头跑全流程
python scripts/run_pipeline.py --data-dir data_dir/my_task

# 从 CTC 预对齐开始, 覆盖已有
python scripts/run_pipeline.py --skip-to prealign --overwrite

# 只重跑后处理 (调参常用)
python scripts/run_pipeline.py --step postprocess --overwrite

# 跳过 trim (音频已预处理)
python scripts/run_pipeline.py --skip-trim

# 在外部目录跑, 不污染项目 workspace
python scripts/run_pipeline.py --data-dir /external/data --output-dir /external/out
```

## 独立工具

### NVV 标注 (`annotate_nvv.py`)

独立的副语言事件检测工具，不跑完整管线:

```bash
python scripts/annotate_nvv.py --input_dir data_dir/my_task --output_dir output/nvv
```

输出: `nvv_annotations.jsonl` (逐文件标注), `summary.txt` (统计), `transcripts_clean.txt`

### Praat 可视化 (`view_in_praat.py`)

匹配 TextGrid + 音频用 Praat 打开:

```bash
python scripts/view_in_praat.py                   # 浏览 output/ 中的 TextGrid
python scripts/view_in_praat.py --dir filtered    # 浏览 filtered/ 中的 TextGrid
python scripts/view_in_praat.py --dir aligned     # 浏览 aligned/
```

### TextGrid 最终清理 (`finalize_textgrids.py`)

对所有 TextGrid 做最终规范化处理（NVV 括号、`<sp1>` 规范化）：

```bash
python scripts/finalize_textgrids.py --input-dir output/ --filtered-dir filtered/ --output-dir finalized/
```

### IPA 映射验证 (`verify_mapping.py`)

验证 IPA 到拼音的映射是否正确:

```bash
python scripts/verify_mapping.py
```

### 英文 Token 词典维护 (`add_english_to_dict.py`)

扫描 CTC 输出中的英文 token 并添加到 MFA 词典:

```bash
python scripts/add_english_to_dict.py --root <ctc_output_dir> --dict dict/mfa_ipa.dict
python scripts/add_english_to_dict.py --root <path> --dict <path> --dry-run  # 预览模式
```

## 环境说明

### MFA 环境 (`mfa_chinese`)

由 `environment.yml` 定义，包含 MFA 3.3.9 + 全部运行时依赖:

| 核心包 | 版本 | 用途 |
|--------|------|------|
| Montreal_Forced_Aligner | 3.3.9 | 声学模型强制对齐 |
| pypinyin | 0.55.0 | 中文→拼音转换 |
| soundfile | 0.13.1 | WAV 读写 |
| kalpy-kaldi | 0.9.0 | Kaldi 绑定 (MFA 内部) |
| praatio | 6.2.2 | TextGrid 读写 |

### NVASR 环境

CTC 预对齐需要额外的 `funasr` + `torch` 环境。通常用 base conda 环境即可 (如果已装 torch)。在配置中设置 `ctc_prealign.python` 指向该环境。

## 常见问题

**MFA 报 "dictionary OOV"**: 检查文本中是否有非 CJK 字符，或标点是否在 MFA 词典中。可运行 `python scripts/verify_mapping.py` 验证 IPA→拼音映射。

**CTC token 数与 MFA word 数不匹配**: 正常现象。`_snap_to_ctc` 会跳过并输出 `stderr` 警告。检查参考文本是否与音频一致。

**后处理质检通过率低**: 先用 `view_in_praat.py` 抽查 aligned/ 中的原始 MFA 输出，判断是 MFA 对齐问题还是 CTC 锚点问题。调整 `ctc_prealign.nvv_bias` 或 MFA 的 beam 参数。

**GPU 显存不足**: 减小 `batch_size_s` (CTC 预对齐时设置，默认 300)。或设置 `device: cpu` 用 CPU 推理。

**Python 找不到**: 在配置中显式设置 `python_path` / `ctc_prealign.python` 为对应 conda env 的 python 路径。空字符串会触发自动搜索。
