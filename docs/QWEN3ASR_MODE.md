# Qwen3-ASR 模式：transcript_only 与 anchored_nvv

日英 TTS 使用独立的 `scripts/run_ja_en_pipeline.py` 入口和
`docs/JA_EN_PIPELINE.md`，不会把 Qwen 中文 profile、中文 TextGrid 或其缓存
当成日英生产证据。日英管线的 ASR family vote、Japanese/English MFA inventory、
Julius diagnostic namespace 和独立 verifier 均单独记录。

## 主管线默认：原生 Qwen 后端与固定时间戳正则化

当前设计见 [Qwen3 主管线设计](superpowers/specs/2026-09-14-qwen3-main-pipeline-design.md)。
无参考文本先识别再生成时间戳，有参考文本只使用 ForcedAligner。两者都将非开头 sp0 并入前词；
sp1/2/3 优先保留原标点并用标点覆盖静音，无标点才补省略号，开头静音不补省略号。
再执行原有 MFA 锚点修正和对齐。最终文本及独立审计采用修复后的文本。
新主流程不会回退 FunASR/NVASR；下文独立 `qwen3asr/anchored_nvv` 属于保留的历史独立工具。

与下文的独立 `mode: qwen3asr` 不同，`ctc_prealign.provider: qwen3_hf` 使用普通
`full` 管线的预对齐位置。它替换该位置上的 NVASR/FunASR 推理，并继续执行 MFA。
配置模板见 `configs/qwen3_hf_canary.yaml`，能力依据见 `docs/QWEN3_UPGRADE_ANALYSIS.md`。

需要两个本地原生模型目录：`Qwen3-ASR-1.7B-hf`（也可选 `0.6B-hf`）与
`Qwen3-ForcedAligner-0.6B-hf`。使用独立环境安装 `requirements-qwen3-hf.txt`。
原生 ASR 要求 Transformers 5.13.0 及以上；还必须具备原生 ForcedAligner 处理器接口。
如果安装版本缺少 `prepare_forced_aligner_inputs` 或 `decode_forced_alignment`，按
[官方模型卡](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf)安装支持该模型的
Transformers 源码版本。管线不会自动下载或转换模型。

先将模板中的数据、模型和解释器路径改为实际本地路径。可直接预检而不加载权重或创建输出：

```bash
/path/to/qwen3-hf/bin/python scripts/qwen3_prealign.py \
  --data-dir /path/to/reference-and-wavs --audio-dir /path/to/wavs \
  --output-dir /path/to/fresh-check-output \
  --model-path /local/models/Qwen3-ASR-1.7B-hf \
  --forced-aligner-model-path /local/models/Qwen3-ForcedAligner-0.6B-hf \
  --reference-mode auto --language Chinese --no-nvv --check
```

预检验证模型树、运行时 API、输入选择及长度；处理器和权重实际加载是否成功仍需短音频
试跑。准备好实际路径后运行 `python scripts/run_pipeline.py --config configs/qwen3_hf_canary.yaml`。
`reference_mode: authority` 对齐已有参考文本；`fallback` 始终识别；`auto` 按每条音频是否
存在非空参考文本选择。可以设置 `language`、`context` 热词提示和 `max_new_tokens`。

当前模型按单条音频离线推理，`batch_size` 控制每卡排队数，`all_gpus` 使用所有可见 GPU，
每卡使用独立的 spawn 进程持有 Qwen ASR 与 ForcedAligner，避免 Transformers 并发加载共享状态。
进程在该卡的全部任务间复用模型，父进程按输入顺序收集结果和写入交接文件。每条音频不超过 300 秒。
显存不足以同时容纳两个模型时，可在 `ctc_prealign` 设置 `device: cuda:1` 和
`forced_aligner_device: cuda:0`，将 ASR 与 ForcedAligner 整体放到不同 GPU。
省略 `forced_aligner_device` 时沿用 `device`；显式设置时要求 `all_gpus: false`。
模型本身的批推理、11 语言对齐、流式 ASR 等能力不代表本次 MFA 管线全部支持这些模式。
现有 MFA 字典交接支持中文字符和英文规范词条；不支持的词条会明确拒绝。

结果包含原有六文件交接，以及参考文本侧车（仅有参考文本时）、manifest、身份记录和
producer/accounting receipts。`provider` 为 `qwen3_hf`，`lexical_timing_source` 为
`qwen3_forced_aligner_hf`。英文参考词保留 canonical metadata；其中历史字段
`source_ctc_ordinals` 表示交接词条的来源序号，另有 `source_ordinal_provider` 标明 Qwen，
不表示模型产生了 CTC 帧。词／字时间戳不能冒充音素时间戳，音素仍由 MFA 生成。

该后端拒绝 NVV 开关及参考文本中的 NVV 事件标签，不生成笑声／喘息等事件候选。
同身份的完整结果可直接复用；失败项可重试；模型、输入、参考文本或已有结果变化时
拒绝继续覆盖，应选择新输出目录。进程中断导致缺少正式身份记录时也要求新目录。

## 目标与非目标

`mode: qwen3asr` 是独立的本地 Qwen producer，不是现有 NVASR/CTC/MFA 管线的新 step。
`profile: transcript_only` 是默认 profile，行为保持不变：只发布原始 Qwen 文本，不产生
时间戳或 NVV。`profile: anchored_nvv` 是显式的中文-only profile，用 Qwen 文本建立词法
锚点，再把 NVASR 的 NVV/标点候选融合回同一条 Qwen 词序。

本模式提供：本地 Transformers 推理、输入和模型身份冻结、逐批原子 checkpoint、失败
重试、篡改检测、manifest 和独立 COMPLETE/PARTIAL receipt。

transcript_only 不提供：ForcedAligner、时间戳、CTC anchor、NVV 插入、MFA、TextGrid、
streaming 发布、自动模型下载、在线 API 或 vLLM。Qwen 文本不会自动复制回源数据目录，
也不会自动成为其他模式的参考文本。anchored_nvv 只增加本地 ForcedAligner、NVASR
candidate timeline 和 provider-neutral fusion；它同样不运行 MFA、TextGrid 或 streaming
发布。

## 组件与生命周期

```text
run_pipeline.py
  ├─ 解析/校验公共配置
  ├─ mode != qwen3asr ──────────────> 原有模式，行为不变
  └─ mode == qwen3asr
       ├─ 按需切换到 qwen3asr.python
       └─ qwen3asr_transcribe.py
            ├─ 冻结 WAV + 模型树 + 推理参数 identity
            ├─ profile=transcript_only → Qwen → transcript artifact
            └─ profile=anchored_nvv
                 Qwen → release → Qwen ForcedAligner → release
                 → NVASR candidate timeline → release → fusion
```

`qwen3asr` 分支位于 MFA Python 查找、普通 workspace 创建、cache scan 和 `STEPS` 执行
之前。完整 resume 在校验模型树、输入和所有已发布 evidence hash 后直接重建审计文件，
不加载 Qwen、ForcedAligner 或 NVASR 权重。

## 现有模式兼容矩阵

| 模式/入口 | 与 Qwen 模式的关系 | 兼容策略 |
|---|---|---|
| `full` | 主管线使用原生 Qwen | 固定时间戳正则化；MFA 与既有后处理保留 |
| `nvrasr_fallback` | 保留模式名，producer 迁移 Qwen | 无参考识别与时间戳正则化，不再调用 NVASR |
| `ctc_ready` | 无关系 | 继续导入既有 CTC bundle，不读取 Qwen output |
| `batch_ctc_ready` | 无关系 | launcher 和 dataset accounting 不变 |
| `strict_replay` | 无关系 | sealed evidence/receipt 不变 |
| `filtered_recovery` | 无关系 | 不把 Qwen 文本作为隐式恢复证据 |
| `mfa_retry` / `mfa_rescue` | 无关系 | retained CTC anchor 契约不变 |
| `streaming_pipeline.py` | 不支持 | 在 GPU 探测、local work/cache 创建前明确拒绝 |
| `qwen3asr` / `transcript_only` | 独立 producer | 不进入 `STEPS`，只发布专属文本 artifact |
| `qwen3asr` / `anchored_nvv` | 独立中文 producer | 不进入 `STEPS`，串行发布 anchor、candidate、fused evidence |

anchored_nvv 已是显式的“Qwen 文本 + 当前 NVV”模式，但只接受 `language: Chinese`，不
把 Qwen 标点当 lexical unit。其他普通模式仍不会因为 `_qwen3.txt` 文件存在就静默采用
Qwen 文本，也不会把 ForcedAligner 时间戳伪装成 NVASR CTC artifact。

## 共享与隔离边界

适当共享的 provider-neutral 能力：

- 公共 YAML/CLI 路径解析和 mode 校验；
- `read_wav_metadata`：统一支持 PCM 与 IEEE-float WAV，并返回物理帧信息和 SHA-256；
- `compute_model_tree_digest`：拒绝符号链接并冻结模型文件树；
- `stable_json_digest`：集合、identity 和 transcript record 的稳定摘要；
- 同目录临时文件加 `os.replace` 的原子写入原则。

必须隔离的 provider-specific 能力：

- `qwen_asr`/`torch` 只在 Qwen 子进程内延迟 import；
- Qwen model factory、language/context、generation 参数和错误类型；
- `qwen3asr-*` checkpoint/manifest/receipt schema；
- transcript-only 输出 namespace；
- resume 和 publication authority。

默认 `requirements.txt` 和 `environment.yml` 不引入 Qwen。可选运行环境由
`requirements-qwen3asr.txt` 单独固定为 `qwen-asr==0.0.6`。

## transcript_only 配置与运行

```yaml
mode: qwen3asr
data_dir: /path/to/wavs
qwen3asr:
  backend: transformers
  python: /home/user/miniconda3/envs/asr/bin/python
  model_path: /mnt/nvme3/models/Qwen3-ASR-1.7B
  output_dir: output/qwen3asr
  device: cuda:0
  dtype: bfloat16
  language: null
  context: ""
  batch_size: 1
  max_new_tokens: 2048
```

`language: null`、`Auto` 或空字符串都转换为官方 API 的 `None`，即自动语言检测。
`--python` 和 `--device` 分别覆盖配置中的 Python 和 device。

```bash
python scripts/run_pipeline.py --config qwen.yaml --mode qwen3asr --qwen3asr-check
python scripts/run_pipeline.py --config qwen.yaml --mode qwen3asr
```

能力检查会验证解释器、`qwen-asr` 版本、API、模型树、torch dtype 和 CUDA device，但不
加载模型、不创建输出。`--workspace`、step/skip/CTC/force/overwrite 等普通管线参数在
Qwen 模式中被明确拒绝，避免“参数被接受但没有语义”。

## anchored_nvv 配置与运行

该 profile 要求三套已存在的本地、非符号链接模型目录，并且强制使用中文：

```yaml
mode: qwen3asr
data_dir: /path/to/chinese-wavs
qwen3asr:
  profile: anchored_nvv
  backend: transformers
  python: /home/user/miniconda3/envs/asr/bin/python
  model_path: /mnt/nvme3/models/Qwen3-ASR-1.7B
  forced_aligner_model_path: /mnt/nvme3/models/Qwen3-ForcedAligner-0.6B
  nvasr_model_path: /mnt/nvme3/models/Multilingual-NVASR
  output_dir: output/anchored_nvv
  device: cuda:0
  dtype: bfloat16
  language: Chinese
  context: ""
  batch_size: 1
  max_new_tokens: 2048
  nvv_bias: 4.0
  pause_threshold: 8
```

```bash
python scripts/run_pipeline.py --config anchored.yaml --mode qwen3asr \
  --qwen3asr-profile anchored_nvv
```

生命周期严格按阶段完成并释放 provider：所有 pending stem 的 Qwen lexical evidence
先写入 checkpoint，随后才运行 ForcedAligner；所有 anchor 写入后才运行 NVASR；两类
evidence 持久化后才运行 fusion。Qwen 是 lexical authority，`lexical_timing_source` 固定
为 `qwen3_forced_aligner`；candidate 的 `timing_source` 只允许
`nvasr_ctc_free_decode` 或 `nvasr_blank_pause_heuristic`。

candidate timeline 保存整段 duration、按 occurrence 区分的 lexical occurrences、候选
相邻 lexical ordinals、source/kind/token IDs，以及不改写的 raw frame 和 speech-relative
frame/second 坐标。fusion 对全局最优的 monotonic lexical mapping 做一致性检查，支持
`before`、`overlap`、`after`、`inter-anchor` 和合法 utterance edge；真实歧义会拒绝，且
每个 candidate 必须 exactly once 出现在 accepted 或 rejected 中。Qwen lexical units 不会
被 NVASR 文本替换。

## Artifact 和身份契约

transcript_only 输出根只允许：

```text
transcripts/{stem}_qwen3.txt
qwen3asr_checkpoint.json
qwen3asr_manifest.json
.qwen3asr_run_receipt.json
```

run identity 绑定：排序唯一 stem；每个 WAV 的相对路径、大小、SHA-256；模型实体路径、
完整文件树和 tree digest；`qwen-asr` 版本；backend、device、dtype、language、context
digest、batch size 和 max tokens。

checkpoint 必须把 frozen source 精确分成 success/failed。resume 只跳过 path、size 和
SHA-256 均匹配的 success；失败项重试。identity 变化、成功文本缺失/篡改、输出目录出现
未记账文件、checkpoint 缺失或集合不守恒都会在推理前 fail closed。

manifest 保存每个成功文本的语言、路径、大小和 hash，以及结构化失败 code、exception
type 和清理后的 message。receipt 绑定 identity、模型、source/success/failed 集合及摘要、
有序 transcript-record digest、manifest/checkpoint hash、argv 和 UTC 时间。全部成功才是
`COMPLETE`；存在推理失败则为 `PARTIAL` 并返回非零。空输入在加载模型和创建输出前拒绝。

anchored_nvv 使用另一组专用根目录，根目录只允许以下 namespace 和审计文件；Qwen 原文
及 lexical units 保存在 checkpoint record 中，不单独写 transcript sidecar：

```text
anchors/{stem}.qwen3_forced_aligner.json
nvasr_candidates/{stem}.candidate_timeline.json
fused/{stem}.anchored_nvv.json
anchored_nvv_checkpoint.json
anchored_nvv_manifest.json
.anchored_nvv_run_receipt.json
```

identity 同时绑定 Qwen、ForcedAligner、NVASR 三棵模型树、schema/provider/zero-width
policy 版本、冻结 WAV inventory、推理参数和 `nvv_bias`/`pause_threshold`。每个阶段文件
都有 path、size、SHA-256；namespace 中的额外文件、篡改文件、schema/identity 不匹配和
不守恒 source 都 fail closed。完整 resume 会在 provider 初始化前完成这些检查，因此不
加载任何模型；未完成 stem 则从最近的阶段继续。

## 验证与真实 smoke

无网络测试使用 fake backend 覆盖：import isolation、官方参数、PCM/IEEE-float WAV、完整
成功、部分失败、空文本、批量长度错误、失败续跑、完整续跑不加载模型、输入/模型 identity
变化、文本篡改、符号链接、CUDA/dtype 能力检查、Python re-exec、旧模式绕过和 streaming
早拒绝。anchored_nvv 另有 test-only Qwen/ForcedAligner/NVASR providers 和预计算 timeline
覆盖串行 phase、候选守恒、全局 mapping、zero-width 和 resume 校验；这些 fixture 不代表
生产模型或全量数据集的准确率。

2026-08-26 使用官方 `Qwen/Qwen3-ASR-1.7B`、`qwen-asr==0.0.6`、GPU 0 对
`LAria_00571.wav` 完成 transcript_only 真实 smoke：1 source、1 success、0 failed，receipt
为 COMPLETE，检测语言为 Chinese。同 identity 再运行没有加载 checkpoint shards，验证
transcript resume 路径。

Qwen 文本为：

> 还有突然间进来这么多朋友，大家晚上好！哎，人好多，好开心，感觉跟做梦一样。

它提供了较完整的标点，但把当前链路中的开头 `RIA`/相邻内容识别成“还有”。该 lexical
差异保留为 audit evidence；anchored_nvv 按设计将 Qwen 作为 lexical authority，这一观察
不构成使用 anchored_nvv 的阻塞项。

anchored_nvv 的 LAria fresh-GPU 验收同样为 `COMPLETE`：Qwen 的 31 个 lexical units 全部
守恒，产生一个 unique lexical mapping，6 个 candidate 全部 accepted、0 个 rejected。
其中 `Sigh` 的 5.52–5.58s candidate overlay 到 Qwen `哎`（ordinal 17），`大`/`家` 的
ForcedAligner zero-width 原始点按 80ms quantum 向右展开并保留 raw timing adjustment，
terminal ellipsis 也被接受。该结果证明本次固定 LAria 输入上的 artifact、mapping 和候选
守恒契约；不能推广为所有数据集的 ASR 或 NVV 质量结论。
