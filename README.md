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
│   ├── launch_8gpu.py             # 兼容启动器：单个批量流水线，不再分片
│   ├── launch_multi_gpu.sh         # 批量多 GPU 启动器
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

## 批量 / 多 GPU 运行

`mode: batch_ctc_ready` 配置（例如 `configs/batch_all.yaml`）先生成扫描缓存，再由
`streaming_pipeline.py` 作为单个批量调度器运行。不要为同一个批次自行启动多个共享
输出目录的分片进程。

```bash
# 1. 生成或刷新数据集扫描缓存（不开始正式处理）
python scripts/run_pipeline.py --config configs/batch_all.yaml --scan-only

# 2. 推荐入口：一个 GPU/CPU 分阶段流水线
python scripts/streaming_pipeline.py --config configs/batch_all.yaml --pipelined
```

流水线在扫描完成后统一规划资源：GPU worker 与 CPU worker 会按实际 batch 数量收敛，
两个队列受限于缓冲区；每个 MFA（含英语 MFA）进程池的 `num_jobs` 也会被限制在主机
CPU 预算以内。配置或命令行中的较大数值是请求值，不会绕过该上限。

常用资源控制如下：

| 控制项 | 作用 |
|------|------|
| `streaming.num_gpus` / `--gpus N` | GPU worker 数；未指定或为 `0` 时自动检测。 |
| `pipelined.cpu_workers` / `--cpu-workers N` | pipelined 模式的 CPU worker 数；`0` 为自动规划。 |
| `mfa.num_jobs`、`mfa_en.num_jobs` / `--mfa-jobs N`、`--mfa-en-jobs N` | 每个 worker 的 MFA 请求值；资源规划器会按 CPU 预算截断。 |
| `mfa.dither`、`mfa_en.dither` | MFCC dither；默认 `0.0`，确保相同输入和配置的 MFA 边界可重复。生产与单条恢复都会显式传给 MFA。 |
| `streaming.prefetch_buffer`、`streaming.upload_buffer` | 两个有界队列的容量；增大可提高吞吐，但需要更多本地 NVMe 空间。 |

MFA 声学模型元数据通常自带 `dither: 1`。随机 dither 会使同一音频的边界在多次运行间漂移，
甚至影响临界音素 QC；本项目因此把中文、英文、分片和单条恢复命令统一固定为 `0.0`。
如需实验性地使用非零值，必须显式配置，并接受结果不可作为确定性重放证据。

`scripts/launch_multi_gpu.sh` 可封装同一条批量流水线：`--streaming` 时默认启用
`--pipelined`，而 `--no-pipelined` 会改走普通批量路径。

```bash
bash scripts/launch_multi_gpu.sh --config configs/batch_all.yaml --streaming --pipelined
bash scripts/launch_multi_gpu.sh --config configs/batch_all.yaml --streaming --no-pipelined
```

`scripts/launch_8gpu.py` 保留为兼容入口，但只会启动一个 pipelined 批量流水线；它只
接受 `mode: batch_ctc_ready` 配置，并会拒绝 strict `ctc_ready` 配置。可先用
`--dry-run` 检查实际命令：

```bash
python scripts/launch_8gpu.py --config configs/batch_all.yaml --gpus 0,1,2,3 --dry-run
```

## 独立 Qwen3-ASR transcript-only 模式

完整架构、模式兼容矩阵、artifact/resume 契约和真实 smoke 记录见
[`docs/QWEN3ASR_MODE.md`](docs/QWEN3ASR_MODE.md)。

`profile: transcript_only`（默认）是与 CTC/MFA/streaming 隔离的本地转录模式，只支持
`qwen-asr==0.0.6` 的官方 `Qwen3ASRModel.from_pretrained` Transformers 后端。
模型目录必须已存在且不能是符号链接；不会下载模型、调用在线 API、使用 vLLM，
也不产生时间戳。安装可选依赖时使用独立文件：

```bash
pip install -r requirements-qwen3asr.txt
```

配置示例：

```yaml
mode: qwen3asr
data_dir: data_dir/my_task
qwen3asr:
  profile: transcript_only  # default; use anchored_nvv explicitly for Chinese anchors
  model_path: /local/models/Qwen3-ASR
  output_dir: output/qwen3asr
  python: /home/user/miniconda3/envs/asr/bin/python
  device: cuda:0
  dtype: bfloat16
  language: null  # null、Auto、空字符串均使用官方自动语言检测
  context: ""
  batch_size: 1
  max_new_tokens: 2048
```

运行前可只做能力检查（不创建输出）：

```bash
python scripts/run_pipeline.py --config qwen.yaml --mode qwen3asr --qwen3asr-check
```

`--python` 优先于 `qwen3asr.python`；若它不是当前解释器，入口会使用该可执行文件
原样重新执行一次 `run_pipeline.py` 命令。`--device` 同样优先于
`qwen3asr.device`。qwen 模式不接受 `--workspace` 或普通 pipeline step flags。

正式运行会冻结排序后的 WAV stem、相对路径、大小和 SHA-256，并将每个 batch
原子写入检查点。恢复只跳过身份完全匹配且内容未被篡改的成功文本；失败 stem
会重试，输入、模型树、版本或运行参数改变时会 fail closed。专用输出根目录只包含：

```text
transcripts/{stem}_qwen3.txt
qwen3asr_checkpoint.json
qwen3asr_manifest.json
.qwen3asr_run_receipt.json
```

`COMPLETE` 仅表示全部 stem 成功；推理异常或部分结果会完整记录成功/失败分区，
生成 `PARTIAL` receipt 并以非零状态退出。空输入会在加载模型及创建输出前直接拒绝。
`streaming_pipeline.py` 明确拒绝
`qwen3asr`，该模式必须直接从 `run_pipeline.py` 启动。

### anchored_nvv 中文锚定模式

需要把 Qwen 词序作为 lexical authority、同时保留 NVASR 的 NVV/标点候选时，显式设置
`qwen3asr.profile: anchored_nvv`。该 profile 只接受 `language: Chinese`，并要求本地的
Qwen ASR、Qwen ForcedAligner 和 NVASR 模型。阶段严格串行：

```text
Qwen → release → ForcedAligner → release → NVASR → release → fusion
```

配置示例：

```yaml
mode: qwen3asr
data_dir: data_dir/my_chinese_task
qwen3asr:
  profile: anchored_nvv
  model_path: /local/models/Qwen3-ASR-1.7B
  forced_aligner_model_path: /local/models/Qwen3-ForcedAligner-0.6B
  nvasr_model_path: /local/models/Multilingual-NVASR
  output_dir: output/anchored_nvv
  python: /home/user/miniconda3/envs/asr/bin/python
  device: cuda:0
  dtype: bfloat16
  language: Chinese
  nvv_bias: 4.0
  pause_threshold: 8
```

Qwen 文本和 ForcedAligner item 共同确定 lexical 锚点，candidate timeline 保留 duration、
lexical occurrence/邻接、source/kind/token IDs 及 raw/speech frame 坐标。fusion 是全局
monotonic：允许唯一的 before、overlay、after、inter-anchor 和有效 edge，真正歧义会拒绝，
并检查 candidate exactly-once conservation。`lexical_timing_source` 固定为
`qwen3_forced_aligner`；candidate timing source 为 `nvasr_ctc_free_decode` 或
`nvasr_blank_pause_heuristic`。

anchored_nvv 只生成专用的 `anchors/`、`nvasr_candidates/`、`fused/` 以及 checkpoint、
manifest、receipt，不生成普通 transcript sidecar、MFA/TextGrid 或 streaming 输出。模型
树、输入和每个阶段 artifact 都纳入 identity/hash 校验；篡改、schema identity 漂移或额外
文件会 fail closed，完整 resume 在 provider 加载前完成检查且不加载模型。完整说明和
LAria 验收记录见 [`docs/QWEN3ASR_MODE.md`](docs/QWEN3ASR_MODE.md)。

旧的独立脚本 `scripts/merge_ria_tokens.py`、`scripts/run_nvasr_batch.py` 和
`scripts/run_nvasr_batch_v2.py` 已移除；批量运行请使用上述入口。

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

标点没有声学实现 (MFA 会转为 `<eps>`)，但有 CTC 时间戳。CTC 时间戳本身不是词汇
authority；只有 authoritative reference semantic sequence 确认的 occurrence 才能
绑定相邻 lexical owners 的 local gap。无 authority 时 CTC-only 标点只有在它与显式
local silence gap 相交且未穿过 lexical owner 时才写回，否则直接移除并保留
edge/interior silence:

- 词-标点重叠 → 裁剪标点，保护词 (不破坏音素完整性)
- 标点 interval → clip 到已确认 occurrence 的 local gap，不向下一个词或 axis 尾部延伸
- 无 local owner gap 或 malformed/missing authority evidence → fail-closed，不猜测时间
- NVV 前方间隙 <=200ms → 吸收进 NVV (NVV 天然含周围静音，但有标点时跳过)

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

### 如何理解 BREATHING 的 60ms 与报告字段

CTC 的单帧宽度是 60ms。报告中的 60ms 只有在
`raw_frame_count=1` 且 `frame_limited=true` 时，才表示完整的一帧级模型支持；它不证明
生理呼吸的真实包络恰好为 60ms，也不表示管线检测到了完整的生理呼吸边界。核读 NVV
报告时，以 `frame_support_span` 和 `raw_frame_count` 判断模型帧支持，以
`owner_required_segments`/`owner_required_span` 检查最终 owner 是否覆盖所需证据；
`display_span` 是非声学显示几何，`display_is_acoustic_evidence=false` 必须保持不变。
`mapping_selection`、`candidate_id` 与 `nvv_deduplication` 用于确认唯一映射和相邻 NVV
去重 provenance；字段缺失、未知或不一致时应按 rejected/fail-closed 理解，而不是把
显示时长当成声学或生理边界。

## 质检与过滤

后处理阶段对每个 TextGrid 做自动质检，不通过的放入 `filtered/`:

| 规则 | 默认阈值 | 说明 |
|------|----------|------|
| `short_phone` | < 0.015s | 中文音素过短 (对齐失败) |
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
  min_sil_merge_sec: 0.2        # 最终 visual words 短静音 owner 上限；0.5 为硬上限
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

`min_sil_merge_sec`（或 CLI 的 `--merge-max-sil-sec`）只记录 visual owner pass 的诊断
上下文；canonical 内部 SP0/SP1 的 eligibility 不由 `merge_silence` 或该配置上限短路。
SP0 语义上限是 `<200ms`，并包含精确 `200000us` 的 stale-SP0 例外；`200001us` 的 stale
SP0 是 hard veto。SP1 上限为 `<500ms`。owner 优先级固定为 hard structural veto、
edge/terminal、局部 punctuation、唯一 ordinal CTC 完整包含、accepted energy，最后才是
`merged_left` fallback。没有方向（缺音频、零/低/歧义能量、phone ambiguity、NVV-adjacent
SP1 缺失或重复 CTC evidence）统一 `merged_left`；accepted right energy 必须保留
`merged_right` 和 `energy_owner` provenance。局部标点胜过所有 lexical owner，穿过 lexical
word 的宽泛 punctuation span 不得保护 gap。leading/terminal、bare/malformed/mixed、SP2/SP3
仍保持原有 fail-closed 语义。决策会记录 label、原始 duration、effective max、owner
evidence 和 fallback reason；phones、hanzi、pinyin_phones 只从冻结后的 words 单向重建。
有效内部 `lexical–<sp0>–lexical` 使用 `valid_internal_sp0_forward`，在无更强 owner 时
确定性并入左 owner；accepted energy owner 使用 `energy_owner` provenance，no-direction
case 使用 `merged_left_fallback`。所有 visual silence owner 决策提交完成后、processed
geometry freeze 之前，最终保留的内部纯静音 interval 再按 serialized integer microsecond
ticks 规范化标签：`<200ms` 为 `<sp0>`、`[200,500ms)` 为 `<sp1>`、`[500,1500ms)` 为
`<sp2>`、`>=1500ms` 为 `<sp3>`。这一步只改标签，不改区间、owner、tier 数量或过滤原因，
也不重新开启 owner arbitration；leading `<sp1>` convention 保持不变。

### English/reference canonical contract

有参考文本时，reference 原始顺序是唯一 semantic projection：CJK、English、NVV、标点
和 other 分开处理，`<spN>` 不参与语义。English 保留 `surface` 与独立 `unit_id`；
`target1`/`target2` 可以共享 dictionary-facing `alignment_token=target`，但不能互相
消费。`jin1`、`rui4` 等拼音不会被当作 English，K-Pop 规范化只改变 semantic/dictionary
key，不改变显示 surface。

CTC/MFA 的 `target`、`1` 等碎片必须在 authority commit 前按文件顺序组成一个连续、完整
的 owner；真实 `tokens.jsonl` 没有 ordinal 时使用稳定文件顺序补全，不使用 substring
猜测，也不跨 CJK/NVV/标点/另一 English unit。producer 会先加有效 padding，再以 padded
clip duration 判断 `min_segment_dur_ms`；缺失 MFA、空 phones 或跨度不完整仍 fail-closed，
不会回退到 CMU/G2P/equal split。strict/publication audit 验证同一多对一 projection，
缺失标点保持 `missing_allowed`，但 extra、错序、跨边界或 partial fragment 继续过滤。

authority/reference mode 在上述 semantic projection 之前执行一次
`reference-numeral-normalization-v1`。原始 reference 文本保持独立的
`reference_text_original_raw`/SHA-256 provenance；只有内存中的 normalized surface 进入
English、pinyin、hanzi 和 strict/audit projection。小写、独立的 `target1`/`target2`
分别变为 `target一`/`target二`，CTC 中末尾 ASCII fragment `1`/`2` 绑定为中文 numeral，
最终显示/拼音为 `target yi1`/`target er4`，不再把完整 `target1`/`target2` 当作一个
English unit。pinyin tone token（如 `rui4`）、NVV、uppercase identifier 和普通 English
numeric identifier（如 `OK2`、`ABC1`）不会转换；raw CTC lab/tokens 与 pinyin tone digits
也不会被改写。归一化字段会记录 schema、engine、mapping 及 raw/normalized hash。

最终 visual words 的短静音 owner pass 按结构和 owner 类型处理
`lexical–<spN>–lexical` 内部 gap。标签语义固定为：
`<sp0>` `<0.2s`、`<sp1>` `[0.2s, 0.5s)`、`<sp2>` `[0.5s, 1.5s)`、`<sp3>`
`>=1.5s`；候选总时长必须与 `silence_label(duration)` 一致。
`min(min_sil_merge_sec, 0.5)` 只作为诊断上下文；canonical SP0/SP1 不因 merge 开关或
configured max 而短路。唯一 ordinal CTC 完整包含 owner 优先，其次是 accepted energy，
最后才是 `merged_left_fallback`。
有效内部 `<sp0>` 在无更强 owner 时使用 `valid_internal_sp0_forward` 左合并；accepted
energy owner 使用 `energy_owner` provenance；缺音频、低/歧义能量、phone ambiguity 或
NVV-adjacent SP1 缺失/重复 CTC evidence 使用 `merged_left_fallback`。标点胜过 lexical
owner；宽泛 punctuation span 若穿过 lexical word 不得保护 gap。首部静音、edge、sp2/sp3
继续保持 fail-closed。NVV 邻接 `<sp0>` 不改写 NVV 身份、顺序或原始 CTC/MFA 证据；NVV
邻接 `<sp1>` 只有唯一同 ordinal CTC span 完整包含 gap 时才使用
`nvv_adjacent_sp1_ctc_containing_owner`，否则进入 energy 或左 fallback，不再 preserve。
最终 visual snapshot 中已有标点若是最后
非静音 owner，且后面直到 `words_tier.xmax` 只有连续纯静音，则执行
`terminal_punctuation_tail_absorption`；若结构为末词、纯静音、句末标点，则执行
`terminal_punctuation_head_absorption` 将静音并入已有标点。轴末 NVV 后仅剩合法且短于
200ms 的 `<sp0>` 时使用 `terminal_nvv_sp0_absorption`，不扩大到 `<sp1>` 或非末尾 gap；
没有显式局部静音时不会仅按 reference/CTC-only anchor 合成缺失标点；有局部显式静音
owner 时会写回对应 punctuation interval。所有 preserve/merge/fallback 原因写入决策
报告。

最终 `raw_text`/`pinyin` 与 `hanzi`/`pinyin_phones` 一样从冻结后的 words owner 事务性
重建。已知 bare NVV（包括 `Surprise-wa`）先规范为 `<SURPRISE-WA>`，标点审计移除完整
NVV markup 后再比较正文标点，因此标签内部连字符不再制造 punctuation mismatch；普通
词内连字符仍按正文标点保留并接受审计。

对内部 lexical–`<sp0>`–lexical gap，局部 punctuation evidence 与显式静音 gap 同时
存在时，resolver 会把该 gap 写回对应 punctuation interval；宽泛 punctuation span 若
穿过 gap 两侧已有 lexical owner，不得保护该 gap。没有显式静音 gap 时仍不合成标点。
`unknown_sp0_forward` 仅作为旧 report/ledger 的兼容 operation 名保留；当前新决策同时
记录 CTC/energy/fallback owner provenance，不把旧的固定左方向当作 energy 结论。

### Pre-CTC subset denominator

在 `mode: nvrasr_fallback` 或 `full` 且没有既有 `expected_stems` 时，
`ctc_prealign.limit > 0` 是整个 pre-CTC 路由的选择边界：pipeline 先对物理 WAV 的
去重 stem 排序，选择前 N 条并写入 `workspace/pre_ctc_stems.txt`。pad_silence、resample、
CTC prealign、MFA 和 postprocess 都必须消费该 manifest；CTC 不再通过第二次 `--limit`
扫描恢复全量。最终 accounting receipt 同时记录 physical source universe、selected
eligible denominator、排除原因 `pre_ctc_limit` 以及 selection schema。已有
`expected_stems` 或既有 manifest 则保持其冻结分母；不会覆盖或扩大选择集。

### Word-energy evidence contract

`filter_word_energy_ratio` 使用单一的 `word-energy-evidence-v1` 审计：词能量和噪声底
都按完整 10ms frame RMS 计算，阈值严格为 `noise_floor * ratio`，不再附加隐藏的
`10x` 下限。噪声池优先取最终 visual words 中显式纯静音 owner 的完整 frames；没有
可用静音 frames 时才回退到全音频 10ms RMS 的 15th percentile。报告保留
`source/frame_count/noise_floor/ratio/threshold`，并逐 lexical ordinal 记录 final、
premerge lexical core、source-phone、CTC spans、RMS/active-run、merge operation、
lineage 与分类。

分类包括 `energetic`、`silence_merge_dilution`（只诊断，扩张后的 display span 不会
稀释词本体判定）、`true_low_energy`（产生 `word_in_silence`）、
`word_energy_boundary_mismatch` 和 `word_energy_evidence_unresolved`。缺失/歧义 lineage、
无 source owner 或没有有效 CTC evidence 时，source phone 的 active overhang、phone
hole/audio hole 才会升级为 hard reason；有效 CTC `ctc_span` 是词边界锚点，相关越界
与 hole 仍写入 diagnostics 但不直接过滤。派生 phones 不会被当作独立声学证据。English/NVV
本身为 `not_applicable`，但中文 owner 可通过静音跨到最近 English/NVV owner；标点会
截断这种邻接关系。显式 `--no-enable-word-in-silence-filter` 优先于 strict mode，
关闭时仍保留诊断报告但不新增能量过滤原因。能量审计只在 visual silence commit、
freeze/lineage rebuild、strict English/phone owner 之后执行一次，publication audit
随后继续独立执行。

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

## Testing

运行仓库测试（禁用字节码和 pytest 缓存）：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests
```

默认 `pytest` 可能遍历模型资产；仓库配置将测试根限定为 `tests/`，因此
`--collect-only` 也只收集测试目录。`tests/run_*` 是专项 fixture/旧 runner；仅已确认重复的 runner 会移除。

### English v2 provenance and replay

当前 English producer/consumer 契约是 `strict-en-mfa-v2`，且 manifest、ledger、
segment/word evidence 必须绑定 `canonical-english-units-v1`。`strict-en-mfa-v1`
只作为历史产物识别，不能被当作本次运行成功，也不会被原地改写；filtered recovery
和 strict replay 必须在 run-local workspace 重新定位已验证的 v2 文件。English MFA
词典角色同样绑定到本次 run-local dictionary 的 SHA-256，禁止用旧生产目录中的词典
冒充当前 provenance。

无参考（no-reference/fallback）路径没有 English lexical evidence 时仍可通过空的
v2 English manifest；一旦输出包含 English 词，就必须提供完整 canonical binding 和
逐词 ledger/source TextGrid evidence。

最终 display publication 还要求 `words`/`hanzi` 在同一音频轴上形成完整、正时长、无重叠
的 owner partition；axis 对齐残差按 `AXIS_EPS` 处理。小于 30 ms 只是候选上限；只有
source words、CTC lexical spans 与（如有）reference semantic sequence 共同证明是
mechanical frame residual 时才吸收，证据缺失或存在 source silence/标点时保留为
canonical `<spN>` 并由 strict audit 过滤。参考标点只拥有相邻 lexical owners
之间的 local gap，不能把末尾 English/CJK 词延伸到 axis 末端。`strict-en-mfa-v2` consumer
会独立验证 SOS 的 `sos-exact-override-v1`、精确五音序列、run-local dictionary SHA-256
和 phone ordinals；APP 的 `AE1 P` 是负向 canary。历史 v1、缺失、重排或篡改记录均
fail-closed。无 ownership proof 的 gap 不会因数值阈值被静默 publish；interior silence
保留为可审计的 `<spN>`，只有 edge silence 可以关闭 publication axis。上述当前行为由
synthetic/focused tests 验证，历史批次只作为 forensic evidence。

### Authority `ok` 100-stem canary

`configs/hecheng_ria_ok100_authority.selection.json` 是固定的 100 条 authority
选择清单：先在历史只读 `ctc_pretg/{stem}_ref.txt` 中用
`(?<![A-Za-z])ok(?![A-Za-z])` 做大小写不敏感的 ASCII 独立 token 匹配，再从完整
audio/CTC/reference bundle 候选中按固定 seed
`authority-ok100-20260818-v1` 的 `sha256(seed + NUL + stem), stem` 顺序取前 100。
清单中的 `candidate_count`、seed、排序规则和历史 manifest SHA 是可追溯元数据；
配置 `configs/hecheng_ria_ok100_authority.yaml` 的 stems 必须与清单逐项相同，且
`reference_mode: authority`。CTC receipt 绑定的 audio root 是
`/mnt/nvme3/mfa_workspace_54k_fresh/padded_audio`；历史目录只提供 CTC/reference
forensic evidence，不能用其中的 `audio_16k` 替代。workspace/output/filtered 使用
`/mnt/nvme3/mfa_workspace_54k_ok100_authority_validation_v9` 下的新隔离路径；配置的
`require_fresh_workspace: true` 会在 workspace 已存在时 fail-closed，不覆盖旧验证或生产目录。

显式 `ctc_ready.stems`/`stem_range` 在 link 完成后冻结为同一个
`ctx.expected_stems`/accounting denominator，并传给 resample、receipt、MFA axis、
postprocess 和 strict-ok。resample 从完整 `data_dir` 只读取这些精确 stem；源目录可
有额外 WAV，但 workspace `audio_16k` 输出必须严格等于冻结子集。未筛选的普通路径仍
保留完整 source namespace 校验。

先做不启动 MFA/GPU 的准备扫描：

```bash
python scripts/run_pipeline.py --config configs/hecheng_ria_ok100_authority.yaml --scan-only
```

fresh run 完成后逐条审计：

```bash
PYTHONPATH=. python scripts/audit_authority_ok100.py \
  --selection configs/hecheng_ria_ok100_authority.selection.json \
  --run-root /mnt/nvme3/mfa_workspace_54k_ok100_authority_validation_v9/strict_ok_runs/<run_id> \
  --evidence-root /mnt/nvme3/mfa_workspace_54k_ok100_authority_validation_v9 \
  --audio-root /mnt/nvme3/mfa_workspace_54k_fresh/padded_audio \
  --report /tmp/hecheng_ria_ok100_authority.audit.json
```

审计器会重新检查每条 reference 的独立 `ok` token、audio/CTC/reference bundle、
MFA aligned evidence、words/hanzi/phone ownership、CTC lexical spans、标点 local
gap、interior SP 以及 English provenance；缺失或不一致只允许进入 `filtered`，并保留
结构化 reason。`output` 与 `filtered` 必须对 100 条恰好守恒，任何“发布但证据不一致”
都会使审计失败。历史 54k 路径仅用于选择清单和 forensic 对照，不是当前 fresh
publication 结果；本项目不以此命令宣称已重跑生产 MFA/GPU/NVASR。

权威 English 单位仍区分 surface identity 与 MFA dictionary key；普通 English reference
中的 `target1`、`target2` 可各自保留为有序 `en-uXXXX` surface unit，并共享可查字典的
`alignment_token=target`。但经过 authority numeral normalization 的
`target一`/`target二` 会把 numeral suffix 作为独立 CJK semantic owner；CTC 的 `1`/`2`
只按 ordered fragment evidence 映射为 `yi1`/`er4`，不能回退成 `target1` English unit。
有限拼音词表中的 `jin1`、`rui4` 等始终不是 English 单位。缺少 reference、CTC lexical
span 或完整 phone provenance 时保持 fail-closed，不会用模糊拼接补齐。

### CTC raw/work/processed 三段契约

CTC 输入分为三个不可混淆的阶段：`ctc_pretg/` 是 producer-owned 的 immutable
raw namespace，必须由 `.ctc_raw_manifest.json` 绑定每个 stem 的六类 artifact、producer
receipt、文件 SHA-256 和 manifest identity；`ctc_pretg_adj/` 是可变的 processed/work
副本，只能由物理 copy 产生，并由 `.ctc_work_receipt.json` 记录 raw manifest path、
raw digest、raw identity、work identity 和 transform/operation lineage。raw 与 work
即使内容相同也不能共享 inode 或 symlink；raw artifact、manifest 或 work receipt 的
digest/identity 不一致时，独立 audit 对整个候选集 fail-closed。

postprocess report 必须同时保存 `ctc_lifecycle.raw_manifest` 和
`ctc_lifecycle.work_receipt` 的 path/SHA-256/identity，以及
`processed_geometry_digest`、`processed_geometry.frozen` 和
`processed_operation_ledger`。最终 `words` tier 当前写出的 interval geometry 是唯一
publication authority；audit 会把它重新解析并与 report 的 lexical published-span
proof 和 geometry digest 逐项绑定。`resolved_span`、raw CTC span 以及早期 MFA span
只能作为历史证据，不能被重新当作 publication authority。最终 words/hanzi 必须仍是
完整、无重叠的 processed frozen geometry；不确定的 owner、gap、punctuation 或 phone
lineage 只能进入 `filtered`。

因此，`ctc_pretg=immutable raw`、`ctc_pretg_adj=mutable processed/work`，而最终
`words=processed frozen geometry` 是严格的阶段边界，不是目录命名约定。旧的单目录
fixture 若没有任何 raw/work marker 仍保留兼容读取；一旦出现任一 marker，audit 要求
整条 raw → work → processed lineage 完整存在。

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
