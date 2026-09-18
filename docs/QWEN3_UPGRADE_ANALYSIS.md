# Qwen3 识别与对齐升级分析

核对日期：2026-09-14。本文记录能力边界和升级依据，不代表真实模型验收已完成。

## 当前仓库实际使用什么

- 现有生产配置使用 `nvrasr_fallback`，由 Multilingual-NVASR / FunASR 提供识别与 CTC 预对齐，随后由 MFA 提供音素对齐。
- `mode: qwen3asr` 是已有独立入口。`transcript_only` 只产生文本；`anchored_nvv` 串行调用 Qwen ASR、旧版 Qwen ForcedAligner、NVASR，再融合非语音事件候选。
- 仓库的 Qwen 示例使用 **Qwen3-ASR-1.7B** 和 **Qwen3-ForcedAligner-0.6B**。识别模型另有 0.6B 规格，大小通过模型路径选择。

## 新模型能对齐到哪一层

| 层级 | 官方能力与本项目含义 |
| --- | --- |
| 英文词 | 支持词的开始、结束时间。 |
| 中文字 | 标准处理器按 CJK 字符输出；中文语言学意义上的分词需另外聚合。 |
| 句子、段落 | 技术报告描述可调整文本单元粒度；本次标准接口集成以词／字时间戳为基础。 |
| 音素 | 没有公开的音素时间戳接口或音素对齐能力承诺，不能替代 MFA 音素层。 |
| 笑声、喘息等事件 | ForcedAligner 不提供现有 NVASR 的 NVV 事件分类输出。 |

官方处理器的时间戳类别默认间隔为 80 毫秒，这是离散表示的间隔，不是误差上限，也不意味着具有音素级精度。把一个词拆成 IPA／拼音，或者在词区间内均分音素，只能得到推测，不能称为声学音素对齐。

## 可利用的功能

识别模型支持自动语言识别、指定语言、上下文／热词提示、批量转录；官方模型家族覆盖 30 种语言和 22 种中文方言，并提供离线及流式方案。官方工具包中的流式路径与本次离线管线接入是不同功能。

ForcedAligner 可以直接对齐已有参考文本，也可以对齐任意 ASR 产生的文本；标准输出包含 `text`、`start_time`、`end_time`。它覆盖 11 种语言：中文、英语、粤语、法语、德语、意大利语、日语、韩语、葡萄牙语、俄语和西班牙语。官方支持最长 300 秒的语音及跨语言场景；更长音频需要可靠的音频和文本分段，不能仅按文本长度分摊时间。模型通过非自回归前向计算预测时间戳，支持批量推理，原生 Transformers 实现可使用 `torch.compile`。

`-hf` 表示原生 Transformers 适配的检查点，不能据此认为它新增了音素识别能力。新的模型目录和原有 `qwen-asr==0.0.6` 加载路径应明确区分。

## 本次接入决策

采用显式 `ctc_prealign.provider: qwen3_hf`，让 Qwen ASR 与 Qwen ForcedAligner 替换新路径上的 NVASR／FunASR 识别和词／字预对齐职责。已有参考文本保持文本权威性；无参考文本先识别再对齐。保持六文件预对齐交接格式和现有 MFA 音素阶段，结果记录真实 provider，不伪造 CTC 帧或音素时间戳。

新路径不生成 NVV，启用相关选项时应明确拒绝。已有 `anchored_nvv` 保留为需要旧事件候选时的兼容路径。

本段最初还写了「旧生产配置不隐式切换，避免将未知精度的新结果混入现有批任务」。该约束随后被撤销：`migrate_main_prealign_config` 现在会对 `mode` 为 `full` 或 `nvrasr_fallback` 的配置主动改写成 Qwen3，详见下方「实施边界」的更正说明。`mode` 本身不变，变的是 `ctc_prealign.provider` 及其配套字段。

## 运行条件与验收

检查时本机已有旧格式 1.7B ASR 和 0.6B aligner 模型目录，尚未找到 `Qwen3-ForcedAligner-0.6B-hf`。原生路径应分别配置 `Qwen3-ASR-1.7B-hf`（或 `0.6B-hf`）与 `Qwen3-ForcedAligner-0.6B-hf`，不能把旧目录改名视为格式转换。共享 ASR 环境的 Transformers 为 4.57.6，低于官方原生 ASR 模型卡要求的 5.13.0。应为原生后端配置独立环境，检查所需处理器与 AutoModel 接口，不原地升级旧 FunASR 环境。

代码测试应覆盖：原生 API 调用、参考文本优先、英文词与中文字顺序、时间区间合法性、六文件交接、provider／模型身份变更、断点续跑、NVV 拒绝和旧路径回归。

生产切换前仍需用实际 `-hf` 权重，在全新输出目录上分别运行一条有参考文本和一条无参考文本的短音频，检查词／字区间、MFA 音素层以及结果溯源。单元测试通过不代表模型精度已经验收。

## 实施边界（合并自 `superpowers/plans/2026-09-14-qwen3-hf-upgrade.md`）

> 该计划写作时称「保留 legacy `nvasr` 作为默认 provider」。此说法已失效：
> `scripts/run_pipeline.py:799` 的 `migrate_main_prealign_config` 在 `mode` 为
> `full` 或 `nvasr_fallback` 时，先把 provider 默认值取为 `qwen3_hf`，再将
> `nvasr`／`funasr`／`qwen3`／`qwen3_hf` 四者一律归一为 `qwen3_hf`。同一函数还会把
> NVASR 的 `model_path` 与 ASR 解释器替换为 Qwen3 对应值，把 `nvv_enabled` 和
> `reference_nvv_enabled` 置 false，并删除 NVASR 专用的 `nvv_bias` 与
> `pause_threshold`。保留的 CTC／replay 输入维持其历史 provider；MFA 段落从不改写。

**生产者契约** — 生产者接收管线当前的 `--audio-dir`、参考文本根、输出根和冻结的 stem
选择器。参考权威 stem 跳过 ASR 推理，直接对齐给定文本；无参考 stem 先跑 Qwen ASR，
再对齐其转写。两条路径都产出下游归一化、边界修正与 MFA 所需的六件 CTC 产物。该
provider 拒绝 NVV 标志。

**环境与运行参数** — 原生环境隔离在 `requirements-qwen3-hf.txt`，旧的
`qwen-asr==0.0.6` 环境不原地升级。canary 配置指向两棵转换后的 `-hf` 模型树，并使用
`batch_size: 1`，因为当前管线每次生产者调用只喂一条音频。

**验证边界** — 生产者拒绝缺失的模型树、非有限或非正的词／字区间，以及超过
ForcedAligner 五分钟上限的音频。模型树、运行时、设置、输入／参考与输出产物身份都会
留存以做断点续跑校验。单元测试使用注入的假后端，不下载权重、不执行生产运行。

## 官方资料

- [Qwen3-ASR 官方仓库及 ForcedAligner 用法](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ForcedAligner-0.6B-hf 模型卡](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf)
- [Qwen3-ASR-1.7B-hf 模型卡](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)
- [Qwen3-ASR 技术报告，第 3 节](https://arxiv.org/html/2601.21337v1)
- [Transformers 官方处理器源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_asr/processing_qwen3_asr.py)
