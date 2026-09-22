# 日语／英语 ASR + MFA 混合对齐管线实施交接

## 元数据

- 生成时间（UTC）：2026-09-21T03:50:56Z
- 仓库：`/mnt/local_E/MFA_Pause/repo`
- Git revision：`ff2e2f34088e61ba45f08d29d32e6f2dd3c72a44`
- 分支：`codex/0915all-full-corpus`
- 任务 slug：`japanese-english-asr-mfa-pipeline`
- 本交接：`handoffs/20260921T035056Z-japanese-english-asr-mfa-pipeline.md`
- 规划路由：root 已显式派发 `gpt-5.6-sol`、`reasoning_effort=high`；本窗口只规划，不实现。
- 检查时工作树已有用户改动：`scripts/qwen3_timestamp_normalization.py`、`tests/test_reference_macro_resolutions.py` 为已修改文件；`configs/qwen3asr_macro_probe.yaml`、`scripts/retire_macro_source_texts.py` 为未跟踪文件。本方案不得覆盖、清理或借用这些改动。

## 目标

在当前仓库内实现一条可单独运行、可断点恢复、可审计的日语／英语 ASR + MFA 管线。它必须处理纯日语、纯英语和同一句内日英混合语音，最终给出词和音素的全局时间；日语另保存 token、读音、mora（拍）与音素的关系。第一版不承担 TTS 训练、F0 实测或重音预测训练。

推荐新增独立入口 `scripts/run_ja_en_pipeline.py`，复用当前仓库已经验证过的 Qwen 模型身份、音频轴收据、MFA 子进程隔离和 strict English ledger 思路。不要把日英模式塞进当前中文 `FULL_STEP_ORDER`：现有 `run_pipeline.py` 仍以 `mandarin_mfa` 和拼音词典为主，日英路由的文本层、词典和后处理语义不同。独立入口能减少中文生产路径的回归面，同时仍共享基础设施。

## 给不熟悉日语的实现者：先区分 mora、音素和普通话音节

日语对齐不能直接套普通话“一个汉字约等于一个带声调音节”的直觉。

- **音素 phone** 是声学模型对齐的最小标签，例如辅音、元音、长辅音或静音。实际符号必须以所用 MFA 模型包的 `phones.txt`／元数据为准。
- **mora（拍）** 是日语的节拍单位。`学校` 常读作 `ガッコー`，可分成 `ガ｜ッ｜コ｜ー` 四拍；促音 `ッ`、长音 `ー`、拨音 `ン` 都各占一拍。mora 与 phone 不是固定一对一，长元音、长辅音及模型词典写法都可能让一个 phone 关联多拍，或一拍关联多个 phone。
- **普通话音节与声调** 常写成类似 `ma1` 的“音节 + 词汇声调”。日语的高低型与重音核不是每个 mora 都自带的普通话式声调；元音清化也不能当成静音。第一版只对齐实际 phone，并保留未来韵律字段的空接口，不推断 H/L。

混合例子：`今日は game をする` 可译为“今天玩 game”。若录音把 `game` 说成英语 `/geɪm/`，该 token 路由给英语模型；若说成日语外来词 `ゲーム`，即使原文仍写拉丁字母，也路由给日语模型。`AIを使う`（“使用 AI”）中的 `AI` 常按日语字母名 `エーアイ` 读，不能只看到 ASCII 就判成真英语。无法从文本或明确证据判断时，状态必须是 `unresolved`，不得猜一个读音后发布。

## 当前仓库行为与可复用能力

### 代码事实

1. 当前主配置默认是中文声学模型 `mandarin_mfa`、IPA 拼音词典和拼音源词典；另有独立 `mfa_en` 配置，默认英语模型为 `english_us_arpa`、词典为 CMUdict、G2P 为 `english_us_arpa`（`scripts/run_pipeline.py:577-595`、`scripts/run_pipeline.py:665-692`）。
2. 当前“中英混合”并非一个双语 MFA 模型。它先从 CTC `words` tier 找出 canonical English units，按连续性和最大间隔组成英语片段（`scripts/align_english_mfa.py:977-1098`）；再从全局 WAV 裁出带 padding 的片段并写英语 `.lab`（`scripts/align_english_mfa.py:1101-1228`）；最后单独执行 `mfa align`，使用英语模型和 ARPABET 词典（`scripts/align_english_mfa.py:1520-1563`）。
3. 英语 OOV 由 run-local 组合词典处理，基础词典不会被原地修改；当前 cache 仅由 OOV 词表摘要命名（`scripts/align_english_mfa.py:1314-1464`）。新日英实现要把模型、基础词典、G2P 和选择策略也纳入签名，避免同词表复用错误证据。
4. strict English 后处理逐词校验身份、顺序、MFA word 和每个 phone 的时间／ordinal，然后才注入统一 phone tier（`scripts/postprocess_textgrids.py:16910-17014`）。较旧的非 strict 辅助路径会按比例缩放、查 CMUdict，甚至在 phone 数不同的时候切分时间片（`scripts/postprocess_textgrids.py:17017-17145`）；日英新路径只能复用 strict 证据原则，不能复用这段启发式重标时逻辑。
5. 混合输出已有 `en:` phone 前缀和英语 ARPABET 判定（`scripts/pipeline_utils.py:1350-1352`、`scripts/pipeline_utils.py:2178-2215`）。日英新输出可采用 `ja:`／`en:` namespace，但不能把现有 IPA→ARPABET 或 IPA→拼音表当作日语映射。
6. Qwen prealign 已明确把 Qwen ForcedAligner 定位为“词级跨度来源”，MFA 仍负责 phone 对齐（`scripts/qwen3_prealign.py:1-7`）。其产物和模型树已有 identity、hash、原子写入和 receipt 思路（`scripts/qwen3_prealign.py:740-784`）。
7. 独立 Qwen transcript profile 可把空串或 `auto` 归一成 `language=None`，但现有 `anchored_nvv` profile 强制 `language=Chinese`（`scripts/qwen3asr_transcribe.py:191-247`）；后者不能直接拿来跑日英。ForcedAligner adapter 接收显式语言参数（`scripts/qwen3asr_transcribe.py:329-339`），transcript-only 刻意不请求时间戳（`scripts/qwen3asr_transcribe.py:606-615`）。
8. 当前运行依赖固定 MFA `3.3.9`（`environment.yml:11-31`、`requirements.txt:9-37`），模型下载器已管理中文、英语 MFA 模型和 Qwen 模型，但没有日语模型（`scripts/download_models.py:53-69`）。
9. MFA 调用已有 run-local `MFA_ROOT_DIR`、`NUMBA_CACHE_DIR`、零 dither 和参数能力预检；相关回归位于 `tests/test_mfa_runtime_capabilities.py:35-104`、`tests/test_run_pipeline_mfa_root.py:14-71`。大批量 shard 还校验源音频链接目标和 hash（`tests/test_mfa_sharded_axis_links.py:8-40`）。
10. 当前 canonical English unit 只承认受控 ASCII 语法，并排除拼音和 CJK（`scripts/english_units.py:23-36`、`scripts/english_units.py:65-111`）；日语 token 不应硬塞入该 schema，应建立新的通用 unit schema，再把英语 unit 作为一种绑定。

### 外部模型与前端事实

- **本方案基线日语模型**：Japanese MFA acoustic model **v3.0.0**，模型 ID `japanese_mfa`，GMM-HMM/MFCC；配套 Japanese MFA dictionary **v3.0.0**。固定资产是 [`japanese_mfa.zip`](https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-japanese_mfa-v3.0.0/japanese_mfa.zip)（92,191,596 bytes）和 [`japanese_mfa.dict`](https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-japanese_mfa-v3.0.0/japanese_mfa.dict)（21,264,022 bytes）。上游未给 digest，下载后必须本地计算 SHA-256 并写入模型收据，不能在代码里编造。
- 选择 v3.0.0 的理由：仓库现有 MFA 3.3.9 和 `mfa align CORPUS DICTIONARY ACOUSTIC_MODEL OUTPUT` 是 legacy 路径。v3.0.0 可作为固定、可复现的第一版资产，并需通过实际 runtime smoke 才算兼容。参考 [Japanese MFA acoustic model v3.0.0](https://mfa-models.readthedocs.io/en/latest/acoustic/Japanese/Japanese%20MFA%20acoustic%20model%20v3_0_0.html)。
- 当前 Hugging Face Japanese 模型在固定 revision `da71f868eaf4f4ebbf4b1941031df192d9e498de` 的元数据标为 v3.3.0、GMM-HMM、16 kHz、MFCC、10 ms frame shift；它属于 MFA 3.4 引入的 HF 目录包／`align_hf` 路线。模型卡写 78 phones，而该 revision 的元数据实际列 83 项，因此清单必须从下载包 `meta.json`／`phones.txt` 读取。它是后续评测升级项，不是第一版基线，不能声称 v3.0.0 是最新版。参考 [固定 revision 元数据](https://huggingface.co/MontrealCorpusTools/japanese_mfa/blob/da71f868eaf4f4ebbf4b1941031df192d9e498de/acoustic/meta.json) 和 [MFA 模型分发版本说明](https://montreal-forced-aligner.readthedocs.io/en/stable/user_guide/models/model_versions.html)。
- **英语模型**继续使用 English (US) ARPA acoustic model **v3.0.0** 和配套字典 v3.0.0，GMM-HMM/MFCC。参考 [English (US) ARPA acoustic model v3.0.0](https://mfa-models.readthedocs.io/en/latest/acoustic/English/English%20%28US%29%20ARPA%20acoustic%20model%20v3_0_0.html)。
- MFA 官方明确：给预训练模型新增词条时不能引入模型未训练的新 phones。英语含 `æ ə θ ð ɹ` 等 phone，日语包没有这些；所以 `japanese_mfa` 不能靠拼接英语词典变成双语模型。
- Qwen3-ASR-1.7B 官方说明支持日语和英语，`language=None` 可自动识别；Qwen3-ForcedAligner-0.6B 支持日语、英语等 11 种语言并给出词／字符时间。官方未保证同句日英 code-switch 的边界准确率，因此其时间只能作为候选锚点，必须过真实 canary。参考 [Qwen3-ASR 官方仓库](https://github.com/QwenLM/Qwen3-ASR)。
- `pyopenjtalk-plus` 的 `g2p_mapping()`／`make_phoneme_mapping()` 能返回形态素—phone 映射，`SurfacePhonemeMapping` 包含 reading、重音核、重音短语和词性；`char_span` 从 `v0.4.1-post9` 起存在。实现需固定该版本/API 并做 capability smoke，不能假设任意 `pyopenjtalk` 都有同一接口。参考 [pyopenjtalk-plus 官方仓库](https://github.com/tsukumijima/pyopenjtalk-plus)。

## 事实、假设、决策和开放问题

### 事实

- 当前中英能力是“主语言 MFA + 抽出英语片段单独 MFA + strict ledger 回填”，不是原生双语声学模型。
- 现有 Qwen transcript、模型树摘要、收据和 MFA 运行隔离可以复用；中文专用 normalizer、拼音映射、`anchored_nvv` profile 不能直接复用。
- 相同汉字的台本与 ASR 文本不能证明录音采用同一读音；若二者都经同一个 G2P，得到相同 reading 也不是两份独立证据。
- MFA 只回答“指定发音在音频上的最佳边界”，不能把输出 phone 序列当成第三个独立 ASR 投票，也不能可靠判断台本发音是否正确。

### 假设

- 第一版输入以单声道或可确定性转成单声道的 WAV 为主，并生成一份 16 kHz MFA 轴；源音频只读。
- 数据方可为少量拉丁 token 提供 `language_override`／`reading_override`，并接受歧义项被过滤，而不是强制产出伪精确 phone。
- Qwen dual-pass mixed canary 至少能给一部分日英相邻 token 提供单调、非负且双 pass 分歧在阈值内的候选跨度。若假设不成立，纯日语／纯英语仍可发布，混合发布保持关闭。

### 架构决策

1. 新建独立入口和 schema，不改变现有中文默认步骤顺序。
2. 全句 Qwen ASR／ForcedAligner 只负责文本候选和粗锚点；日语、英语相邻 runs 各自裁带上下文的音频片段，由各自 MFA 模型完成 phone 对齐。
3. 日语基线固定 `japanese_mfa` acoustic v3.0.0 + dictionary v3.0.0；英语固定 `english_us_arpa` v3.0.0。HF Japanese v3.3.0 只作为显式升级实验。
4. 所有 corpus token 都写入 run-local **单发音锁定词典**。MFA transcript 不直接使用 surface，而使用纯 ASCII、逐 unit 唯一的 alias，例如 `ju_000012`／`eu_000013`；词典以 alias 为词头并且恰好一条 pronunciation，ledger 保存 `alias -> unit_id -> surface -> selected reading/pronunciation` 反向映射。这样同一 run 内相同 surface 的不同 reading 不会互相覆盖。IV/OOV 执行同一流程；日语锁定 reading→MFA phone 序列，英语锁定 CMUdict／override 选中的 ARPABET 序列。
5. `orig_text`、`display_text`、`spoken_text`、`comparison_text` 和 `reading` 分层保存。Ruby、数字、占位符只能由有 provenance 的规则转换，不能无条件删除。
6. 混合输出保留 native phone，并增加 `ja:`／`en:` 输出前缀和 `language` 字段；不把日语 IPA 映成英语 ARPABET，也不调用现有中文 IPA→拼音表。
7. 语言 seam 上的 phone 只可通过扩大上下文、重估锚点并双侧重跑来解决。若仍越过唯一 ownership 区间，拒绝该 seam／utterance；禁止裁短 phone 或按比例重新分配。
8. mora 只保存与 phone 的多对多关系。只有完整映射时可给出“相关 phones 的包络区间”，并标记 `derived_union`；不得把一个词的时长等分到 mora。
9. 能量只帮助选择重试窗口或检查边缘，不把促音闭塞、清化元音识别成静音，也不覆盖 MFA 原始边界。原始对齐与任何修正必须并存。

### 开放问题摘要

- Qwen ForcedAligner 对同句日英 code-switch 的实际单位、语言参数和 seam 误差，需要真实 GPU canary 决定 mixed publish 是否开启。
- 数据集中 `AI`、`USB`、专名、英文借词究竟按日语还是英语读，需要业务 lexicon／人工 override；默认保持 `unresolved`。
- Japanese v3.0.0 资产的本地 SHA-256、实际 phone inventory 和 MFA 3.3.9 兼容性要在模型准备阶段生成证据。
- kana→Japanese MFA phone 的确定性映射必须用配套词典、模型 phone inventory 和人工 golden cases 验证；不能照搬另一项目的表或凭 IPA 常识补符号。

## 范围、非范围和不变量

### 本次实现范围

- Qwen 日／英 transcript 与全句候选锚点；参考文本存在时保留 authority 和 ASR evidence 两路。
- 日语分词、reading 候选、语言路由、单发音锁定词典。
- 双 MFA 分段对齐、全局 sample 轴映射、strict merge、失败账本、缓存／恢复、独立 verifier。
- 纯日语、纯英语、日英同句三类输出及现有中英回归。
- 面向使用者的配置、模型准备、运行和输出说明。

### 非范围

- 多家 ASR 模型的全量部署、TTS 训练、音高 F0 提取、H/L 自动预测和重音模型训练。
- 把 romanized Japanese、借词或字母串自动猜成某一种语言后强行发布。
- 自动修剪源 WAV，或把训练音频和 MFA 音频放在不同、未收据绑定的时间轴。
- 将 Japanese HF v3.3.0 直接替换 legacy v3.0.0；升级必须另建环境和 A/B 结果集。

### 必须保持的不变量

- 源音频、现有中文输出和用户工作树改动只读；所有新运行使用 fresh workspace。
- 时间以整数 sample 为权威，秒数由 `sample / sample_rate` 派生；同一 stem 的所有片段可精确映回唯一音频轴。
- 每个 expected unit、segment、stem 恰好落入 `verified`、`rejected` 或 `unresolved` 一类，集合守恒。
- 词典 phones 是各自模型 inventory 的子集；silence 集另行验证。
- 无完整模型／词典／前端／配置／输入 hash，不得命中 resume cache。
- 日语整句粗锚点中覆盖英文的占位结果不能成为最终日语 phone 边界证据。

## 目标数据流

```text
source WAV + optional reference
  -> immutable audio inventory / 16 kHz transform receipt
  -> Qwen3-ASR transcript evidence (language=None)
  -> Qwen3-ForcedAligner full-utterance candidate anchors
  -> layered text normalization
  -> pyopenjtalk-plus token/readings + override lexicon
  -> language/pronunciation plan
       ja run -> locked Japanese dict -> Japanese MFA v3.0.0
       en run -> locked ARPA dict     -> English MFA v3.0.0
  -> map local sample spans back to global sample axis
  -> seam validator / retry both adjacent runs / reject on conflict
  -> strict per-language ledgers
  -> merged TextGrid + alignment JSONL + run receipt + rejection ledger
  -> independent verifier
```

### 各阶段输入输出

1. **inventory**：读输入 manifest 和 WAV header；写 `audio_inventory.json`、`audio_transform_receipt.json`。若重采样，保存源／目标帧数、sample rate、时长、SHA-256 和确定性参数。
2. **asr_anchor**：写 `asr/{stem}.json`，保留 Qwen 原文、detected language、模型树摘要；写 `anchors/{stem}.json`，保留 ForcedAligner 原始单位、原始秒数、换算 sample、调用语言和模型摘要。纯日语只跑 `language=Japanese`，纯英语只跑 `language=English`；mixed 对完整 `spoken_text` 固定跑 Japanese/English 两次。两次都按下述 dual-pass v1 算法聚合，异常不进入 frontend。
3. **frontend**：写 `plans/{stem}.json`。每个 unit 具有稳定 ID、原文字符跨度、各文本层、language route、reading／pronunciation、来源、候选锚点和状态。
4. **segment**：把相邻同语言 units 合为 run；根据 anchor 加上下文 padding，写精确 `clip_start_sample`／`clip_end_sample` 和 crop hash。无静音切换也必须保留相邻 ownership 约束。
5. **dict**：写 `dicts/{run_id}.{ja|en}.locked.dict`、alias 反向映射、选择账本和 inventory 校验报告。每个 corpus unit 生成纯 ASCII 唯一 alias，每个 alias 在词典中只能有一条本次发音；相同 surface 可因 unit／reading 不同而拥有不同 alias。
6. **mfa_ja / mfa_en**：分别写原始 TextGrid、命令日志、return code、runtime/model/dict digest 和 per-run strict ledger。模型输出永不原地改写。
7. **merge**：将 local sample 映回 global sample；校验顺序、覆盖、gap、overlap 和相邻语言边界；写 `merged/{stem}.TextGrid`、`merged/{stem}.alignment.json`。
8. **verify / publish**：独立重读源 WAV、计划、词典、MFA 原始输出和 merged 输出；只对验证通过的 fresh run 写 `COMPLETE` receipt。其余写 `PARTIAL` 和精确拒绝原因。

### Mixed candidate anchor：`dual_full_utterance_v1`

第一版算法固定如下，实施者不得自行换成“按脚本猜时间”或只运行一种语言：

1. Qwen ASR 对整段音频以 `language=None` 生成 transcript；reference 存在时，reference 和 ASR 仍是两份独立文本证据，由 frontend 决定本次 `spoken_text`，并保存选择原因。
2. frontend 先把 `spoken_text` 投影为无标点的 lexical character stream，同时保存每个规范化字符到原字符半开跨度的可逆映射。每个 unit 已有稳定 `unit_id`、字符跨度和 `route`。
3. 纯日语只对整段 `spoken_text` 调一次 Qwen ForcedAligner，`language=Japanese`；纯英语同理用 `language=English`。
4. mixed 对**同一完整 `spoken_text` 和同一 WAV**固定调用两次：pass J 使用 `language=Japanese`，pass E 使用 `language=English`。两次 raw items、调用顺序、模型 identity 和时间量化信息都保留。
5. 每个 pass 的 item text 经过与第 2 步相同的可逆规范化，然后按字符做 exact monotonic alignment。只有 lexical characters 全部恰好覆盖一次、顺序不倒置、区间有限且在 WAV 内的 pass 才有效；不使用模糊字符串相似度补缺字符。
6. 由字符跨度把每个 pass 聚合到同一组 units：unit start 是覆盖它的首 item start，unit end 是末 item end。`route=ja` 的 run 使用 pass J 的 unit spans，`route=en` 的 run 使用 pass E 的 unit spans；对应 pass 缺失时该 run 为 `anchor_unusable`，不能借另一语言 pass 伪装成功。
7. 对每个语言 seam，分别在 pass J、pass E 中取 `round((left_unit_end + right_unit_start) / 2)` 的整数 sample 边界。任一 pass 在该处倒置、缺 unit 或越 WAV，seam 即不可用；两个边界估计相差超过配置 `anchor_max_disagreement_ms`（基线 80 ms）则记 `anchor_conflict`。两者都有效且分歧不超阈值时，取 `floor((boundary_J + boundary_E) / 2)` 作为初始 ownership seam，并保留两原值。
8. 初始 seam 只决定裁片 ownership／重试搜索中心，不会写入最终 phone。MFA 后若任一 raw phone 侵入邻语言 ownership、贴裁片边或两侧 overlap，必须按相同 transcript 扩大上下文并双侧重跑；仍失败则拒收。

Qwen 的字／词单位与 MFA token 不一致时，第 5-6 步按字符跨度聚合解决；无法 exact cover 的情况显式失败。这一算法是可测试的第一版选择，真实质量由 held-out gate 决定，而不是把“支持日语和英语”解释为已经证明 code-switch 准确。

## 计划 schema 与 CLI

以下都是**计划新增接口**，不是当前可运行命令。

### 输入 manifest `ja-en-input-v1`

每行至少包含：

```json
{
  "stem": "demo_0001",
  "audio_path": "/absolute/path/demo_0001.wav",
  "orig_text": "今日は game をする",
  "speaker": "speaker_01",
  "language_overrides": [{"char_span": [4, 8], "language": "en", "source_id": "review-20260921-001"}],
  "reading_overrides": []
}
```

`orig_text` 可为空，此时 ASR 是 spoken text 来源；有台本时也保留 ASR transcript，不让 ASR 静默覆盖台本。每个 language／reading override 必须绑定半开字符跨度和非空 `source_id`。

### alignment plan `ja-en-alignment-plan-v1`

unit 必须含：`unit_id`、纯 ASCII 唯一 `corpus_alias`、`orig_span`、`orig_text`、`display_text`、`spoken_text`、`comparison_text`、`route`（`ja|en|unresolved`）、`route_source`、`route_source_id`、`reading_ja`、`pronunciation_en`、`pronunciation_source`、`source_token_ids`、`anchor_start_sample`、`anchor_end_sample`、`anchor_source`、`confidence_flags`、`status`。

Latin token 路由优先级固定为：人工／项目 lexicon override > 明确 ASR lexical evidence > 已验证的日语 reading > 明确英语 pronunciation > `unresolved`。文字形态只能生成候选，不能越过前四级直接发布。

### merged alignment `ja-en-alignment-v1`

- `words[]`：全局 sample／秒、unit ID、route、surface、spoken form、source ledger。
- `phones[]`：`phone_id`、`language`、`native_label`、`output_label`、global start/end sample、模型 ID／版本、dictionary entry digest、raw MFA interval ordinal。
- `moras[]`：kana、`phone_ids[]`、`relation`（`exact_sequence|derived_union|unresolved`）；不强制独立时间。
- `segments[]`：clip sample 范围、padding、offset、MFA command identity、retry history。
- `seams[]`：左右 run、候选区间、最终判定、能量诊断、重试次数、拒绝原因。
- `text_layers`、`audio_axis`、`model_assets`、`frontend_identity`、`run_identity`。

TextGrid 输出三层必需 tier：`words`、`phones`、`language`。`phones` 使用 `ja:`／`en:` 前缀；mora 的多对多关系以 JSON 为权威，避免 TextGrid 伪造一对一时间。

### 配置建议

计划新增 `configs/japanese_english_mfa.yaml`，关键字段：

```yaml
pipeline: ja_en
workspace: /fresh/run/root
input_manifest: /data/ja_en_input.jsonl
asr:
  model_path: /models/Qwen3-ASR-1.7B-hf
  forced_aligner_model_path: /models/Qwen3-ForcedAligner-0.6B-hf
  language: null
frontend:
  pyopenjtalk_plus_version: v0.4.1-post9
  route_ambiguous_latin: unresolved
  override_lexicon: /data/ja_en_overrides.jsonl
mixed:
  enabled: false
  anchor_strategy: dual_full_utterance_v1
  anchor_max_disagreement_ms: 80
  heldout_gate:
    mixed_total: 40
    min_mixed_accepted: 34
    min_per_bucket_accepted: 8
    seam_mae_ms_max: 80
    seam_p95_ms_max: 160
mfa:
  runtime_python: /envs/mfa/bin/python
  dither: 0.0
  japanese:
    acoustic_model: /models/mfa/japanese_mfa-v3.0.0.zip
    base_dictionary: /models/mfa/japanese_mfa-v3.0.0.dict
  english:
    acoustic_model: /models/mfa/english_us_arpa-v3.0.0.zip
    base_dictionary: /models/mfa/english_us_arpa-v3.0.0.dict
seam:
  padding_ms: 100
  max_retries: 2
  allow_phone_clipping: false
publish:
  require_all_resolved: true
```

字段值需由 schema 校验；示例路径是操作员需替换的绝对路径，不是默认文件位置。

计划运行命令：

```bash
python scripts/download_models.py --only mfa --include ja-en --check
python scripts/run_ja_en_pipeline.py --config configs/japanese_english_mfa.yaml --check
python scripts/run_ja_en_pipeline.py --config configs/japanese_english_mfa.yaml --stage inventory,asr_anchor,frontend
python scripts/run_ja_en_pipeline.py --config configs/japanese_english_mfa.yaml --stage align,merge,verify
python scripts/verify_ja_en_alignment.py --workspace /fresh/run/root
```

`--check` 只读检查依赖、模型资产、API、phone inventory、配置和输入，不加载大型模型、不创建输出。stage 续跑只接受完全相同的 run identity。

## 影响文件和符号

| 文件／符号 | 当前证据 | 计划变更 |
|---|---|---|
| `scripts/run_ja_en_pipeline.py`（新增） | 当前只有中文 `STEPS`／`FULL_STEP_ORDER`，见 `scripts/run_pipeline.py:6955-6975` | 新增独立编排、fresh workspace、stage graph、集合守恒和 receipt；复用公共 helpers，不注册进中文默认顺序。 |
| `scripts/ja_en_schema.py`（新增） | English 有独立 canonical schema，见 `scripts/english_units.py:23-36` | 定义输入、plan、language run、per-model ledger、merged、receipt 的版本化校验器和原子 JSON/JSONL writer。 |
| `scripts/ja_en_frontend.py`（新增） | 当前 English parser 只处理受控 ASCII，见 `scripts/english_units.py:225-251` | 实现文本分层、pyopenjtalk-plus capability 检查、token/reading、Latin 歧义路由、override 和 unresolved ledger。 |
| `scripts/japanese_pronunciation.py`（新增） | 仓库没有 kana→Japanese MFA inventory 的实现 | 从配套资产读取 inventory，生成并校验确定性 kana→MFA phones；所有 IV/OOV token 单发音锁定；写映射版本摘要。 |
| `scripts/ja_en_anchor.py`（新增） | Qwen ForcedAligner 只承担 lexical spans，见 `scripts/qwen3_prealign.py:1-7` | 抽取 provider-neutral ASR／aligner adapter，保留 raw item 和 sample 映射；新增日英 code-switch 候选锚点和失败分类。 |
| `scripts/align_japanese_mfa.py`（新增） | 当前无 Japanese MFA runner | 建日语分段 corpus、run-local dict、运行／解析 MFA、生成 `strict-ja-mfa-v1` ledger。 |
| `scripts/align_english_mfa.py:build_en_dict/run_en_mfa` | 当前 run-local OOV dict 与隔离执行见 `scripts/align_english_mfa.py:1314-1464`、`1520-1563` | 抽取或新增不破坏旧 CLI 的共享 segment runner；日英入口要求所有英语 token 单发音锁定，旧 strict-en v2 继续兼容。 |
| `scripts/merge_ja_en_mfa.py`（新增） | 当前 strict injection 校验 identity／phone，见 `scripts/postprocess_textgrids.py:16910-17014` | 全局 sample 映射、seam ownership、双侧 retry、拒绝分类、合并 TextGrid／JSON。禁止使用旧比例缩放路径。 |
| `scripts/verify_ja_en_alignment.py`（新增） | 当前有多类独立 verifier 思路 | 使用独立解析路径重算 hash、词典 pronunciation、phone inventory、全局时轴、集合守恒和 namespace。 |
| `scripts/pipeline_utils.py:compute_model_tree_digest/stable_json_digest` | 已有模型树和稳定 JSON 摘要，见 `scripts/pipeline_utils.py:2533-2594` | 只抽取真正语言无关的 atomic/hash/audio-axis helper；现有中文常量和 phonemap 不扩展为日语。 |
| `scripts/download_models.py:MFA_MODELS/download_mfa` | 目前不含日语，见 `scripts/download_models.py:53-69`、`141-169` | 增加显式 `ja-en` 资产组和固定 v3.0.0 URL下载；校验 byte size、计算 SHA-256、写 asset receipt。默认现有下载行为不变。 |
| `requirements-ja-en.txt`（新增） | Qwen 已用独立 requirements，见 `docs/QWEN3ASR_MODE.md:130-132` | 固定 `qwen-asr==0.0.6`、pyopenjtalk-plus 对应 `v0.4.1-post9` 的可安装版本及必要前端依赖；MFA 仍由现有环境提供。 |
| `configs/japanese_english_mfa.yaml`（新增） | 无日英配置 | 给出可审计示例并默认 fail closed。 |
| `README.md`、`docs/QWEN3ASR_MODE.md` | Qwen 模式当前说明 transcript 与中文 anchored profile，见 `docs/QWEN3ASR_MODE.md:93-110`、`133-161` | 写日英入口、模型版本、语言路由、输出解释、限制；明确 `anchored_nvv` 仍仅中文。 |
| `tests/test_ja_en_*.py`（新增） | English canonical／MFA runtime 已有测试，见 `tests/test_align_english_mfa_canonical_units.py:283-387` | 增加 schema、frontend、锁定词典、双模型 runner、sample 轴、seam、resume、独立 verifier 和真实 smoke marker 测试。 |
| 现有回归集 | `tests/test_mfa_runtime_capabilities.py:35-104`、`tests/test_run_pipeline_mfa_root.py:14-71` | 保持全部通过，证明新入口未改变中文／英语生产语义。 |

## 编号需求

**R1. 输入与文本分层。** 输入 manifest、音频 inventory 和五类文本／reading 字段必须版本化、可追溯；源文件不可修改。

**R2. ASR 与候选锚点。** Qwen ASR 和 ForcedAligner 的模型、参数、原始结果、检测语言和时间单位必须单独保留；mixed 必须执行 `dual_full_utterance_v1` 两个语言 pass 和 exact character-span 聚合；锚点不是 MFA phone 证据。

**R3. 语言路由。** 纯日语、纯英语和日英混合均可表示；Latin token 必须按发音意图路由，`AI/USB`、日语化借词、真英语和未知项有不同状态。

**R4. 发音选择与锁定。** 每个 token 必须有唯一选择的 reading／pronunciation 及来源；所有 IV/OOV unit 都获得纯 ASCII 唯一 corpus alias，locked dict 对每个 alias 恰好一条发音并可反查 unit/surface。未知项不可借 MFA 输出补成已知。

**R5. 模型基线和资产 provenance。** 日语使用 `japanese_mfa` v3.0.0 + 同版 dictionary；英语使用 `english_us_arpa` v3.0.0 + 同版 dictionary；runtime、资产 size/hash、metadata 和 inventory 均入 receipt。

**R6. Phone inventory。** 任一词典行出现模型 inventory 外 phone 都在 MFA 启动前失败；日英 phone namespace 不能相互污染。

**R7. 冻结时间轴与分段。** 裁片用整数 sample，local→global 映射可逆；训练／对齐音频必须由 transform receipt 绑定。padding 不改变 lexical ownership。

**R8. 日语对齐。** 日语 run 仅把日语 token 送入 Japanese MFA，并产生 raw TextGrid、严格 ledger 和 token→mora→phone 关系。

**R9. 英语对齐。** 英语 run 复用 English ARPA strict 语义，短词也有明确尝试门槛／拒绝原因；不使用 Japanese MFA 的英语占位 phones。

**R10. 混合 seam。** 相邻语言无静音切换、重叠、gap、单语模型吞邻语音等情况必须经双侧 ownership 校验；失败只可重锚／重跑或拒绝，不能裁 phone。

**R11. 合并输出。** 全局 `words`、`phones`、`language` tier 和 JSONL 必须顺序单调、区间合法、identity 完整；mora 与 phone 多对多，不做强制等分。

**R12. Cache 与恢复。** cache identity 覆盖输入音频、文本、模型树、MFA 资产、词典、前端版本／映射、配置、schema 和代码摘要；篡改、额外文件、签名漂移均 fail closed。

**R13. 失败与集合守恒。** expected stem/unit/run 精确分成 verified、rejected、unresolved；`PARTIAL` 不得伪装 `COMPLETE`，错误需稳定 code 和证据路径。

**R14. 兼容性。** 当前中文主线、strict English v2、MFA root 隔离、shard 音频 hash 和 Qwen 中文 profile 行为不得回归。

**R15. 用户文档。** 文档必须用中文解释日语 phone/mora 与普通话差异、模型选择、日英例子、配置、恢复、输出和局限。

**R16. 韵律扩展接口。** 可预留 accent／F0 mask 字段，但默认未知；不得把 pyopenjtalk 预测重音写成实测 F0 或已验证 H/L。

**R17. 发布门禁。** 无 GPU 测试、模型 capability smoke、真实纯日／纯英／混合 canary 和独立 verifier 全通过后才允许 mixed `COMPLETE`。

## 有序实施计划

### 0. 冻结契约与测试素材

所有者：pipeline owner。依赖：无。

记录当前 revision／dirty status；创建不与现有输出重名的 workspace。准备最小合成 fixture，并冻结一套 60 条、后续调参不可见的人工 held-out：10 条纯日语、10 条纯英语、40 条 mixed。mixed 分四个各 10 条的主桶：日→英、英→日、无停顿切换、单个短英文词紧贴日语助词；各桶内部覆盖 `AI/USB` 日语读法、真英语、发音未知项、单语模型越界和 Qwen 字／词单位与 MFA token 不一致。人工 gold 标 route、spoken reading/pronunciation、关键 word span 和语言 seam sample；不需要标逐 phone 边界。另建非 held-out 开发集调阈值，禁止用 held-out 回调参数。

### 1. 先定义 schema、状态机和独立 verifier 骨架

所有者：schema／audit 实施者。依赖：0。

实现 `ja_en_schema.py`、JSON schema validator、原子 writer、stable error codes 和 exact-set partition。给 verifier 建独立的 TextGrid／JSON 读取路径，避免 producer 与 verifier 共享一个错误。先写正反 fixture：重复 unit、倒置区间、非有限时间、额外 artifact、hash 篡改、集合不守恒均非零。

### 2. 固定模型资产与 runtime capability

所有者：runtime／model 实施者。依赖：1。

扩展下载器的可选 `ja-en` 组，按固定 URL 和 byte size 获取 v3.0.0 日语资产，下载后计算 SHA-256；英语资产也记录实际 hash 和版本。解析 acoustic／dictionary metadata 与 phone inventory。实际调用当前 MFA 3.3.9 依次运行版本检查、最小 `validate`、最小 `align`；任何命令或输出格式不匹配即阻塞真实对齐。另写 HF v3.3.0 capability report，但不接入基线执行。

### 3. 实现 ASR／anchor adapter

所有者：ASR 实施者。依赖：1。

复用 Qwen 模型树 identity、batch／resume 和 `_ForcedAlignerAdapter` 的合法部分，建立独立日英 profile。严格实现 `dual_full_utterance_v1`：ASR 用 `language=None`；纯日／纯英跑对应单 pass；mixed 对相同完整 `spoken_text` 分别跑 Japanese/English pass，经同一 lexical character stream 做 exact monotonic span alignment，按 unit char span 聚合，route 使用对应语言 pass；两个 seam 中点估计取整数均值下界，分歧超过 80 ms 则 `anchor_conflict`。保留所有 raw items 和两原始 seam 估计。强制单调、WAV domain、非负跨度和全 unit coverage；不合格项进入 `anchor_unusable`。

### 4. 实现日语前端和语言／发音决策

所有者：frontend 实施者。依赖：1、3。

固定 pyopenjtalk-plus API capability，输出形态素、reading、POS、char span。建立 reversible normalizer 和五层文本。将 reference、ASR、override、项目 lexicon 的 reading 候选分开保存；相同文本经同一 G2P 得到的相同 reading 只算共享推导，不增加声学置信度。实现 Latin 路由优先级和 `unresolved`。跨 ASR 家族证据接口可保留 `reading_evidence[]`，但第一版不得伪称已部署五模型共识。

### 5. 实现 kana→MFA phone 与单发音词典

所有者：lexicon 实施者。依赖：2、4。

从 Japanese v3.0.0 词典和 inventory 建确定性 mapping，覆盖普通假名、拗音、促音、拨音、长音、清化候选和专名。所有 mapping 都有 golden fixture；不能添加 inventory 外符号。按排序稳定 `unit_id` 生成 `ju_`／`eu_` 加零填充十进制 ordinal 的纯 ASCII alias；同 stem 内不得重复，跨 stem 由 corpus 路径隔离。MFA `.lab` 只写 alias，locked dict 对每个 alias 恰好一行，`alias_map.jsonl` 绑定 unit、surface、selected reading、phone sequence、source 和行 digest。同 surface 不同 reading 因 alias 不同而可并存。英语同样锁唯一 ARPABET pronunciation；多发音且无选择证据时 unresolved。

### 6. 建立 sample 轴、language runs 和裁片

所有者：audio／segmentation 实施者。依赖：3、4。

复用 WAV metadata 与 transform receipt，生成 16 kHz 单声道 MFA 轴；所有边界先量化为 sample。按相邻 route 合并 runs，padding 仅扩音频上下文，ownership 仍按 anchor／邻接约束。裁片写 hash、源轴 offset 和帧数。无静音 seam 两侧都保留上下文，但不能把上下文 phone 作为本 run 所有。

### 7. 日语与英语 MFA runner

所有者：MFA 实施者。依赖：2、5、6。

新增 Japanese runner；抽取 English runner 的语言无关 subprocess、日志、MFA root 和 strict parser，而不改变旧 CLI 输出。每个语言 run 使用各自 acoustic model、locked dict、独立 MFA root/temp。解析 raw TextGrid 时验证期望 token 顺序、每词至少一个合法 phone、phone 属于模型 inventory、区间在 clip 内且不触边到疑似截断。生成 per-run ledger，不直接修改合并输出。

### 8. Seam resolver 与合并

所有者：alignment merge 实施者。依赖：7。

把两侧 raw phone 映回全局 sample，验证唯一 ownership。若 phone 侵入邻语言 core、两侧 raw word/phone 重叠、首尾 phone 贴 clip 边或 Qwen anchor 与 MFA token cardinality 不一致，先扩大／平移上下文并重跑相邻两侧；再尝试字符→词重新聚合锚点。达到 `max_retries` 后拒绝 seam 和相关 utterance。能量只作为诊断／重锚候选，不直接切 phone。通过后按 unit 顺序合并，并保存每次 retry 证据。

### 9. 编排、cache、恢复和发布

所有者：pipeline 实施者。依赖：1-8。

实现独立 stage graph 和 dry check。run identity 包含所有输入和依赖摘要；每 stage 写 checkpoint／manifest，临时文件同目录原子替换。完整 resume 在加载 GPU／MFA 前验证 namespace 和 hash；失败 stage 从最近有效边界继续。最终只在独立 verifier 通过后写 `COMPLETE`；目标 publish 目录必须不存在。

### 10. 文档、回归与真实 canary

所有者：QA／运行责任人。依赖：9。

补 README、Qwen mode 边界和日英操作手册。先跑无 GPU suite 与现有回归，再做模型 smoke；在开发集冻结阈值后只运行一次 held-out。mixed 发布门槛是：40 条 mixed 至少自动接受 34 条；四个主桶各至少接受 8/10，因而日→英、英→日、无停顿和短英文切换都必须有实际成功；accepted seams 相对人工 gold 的 MAE 不高于 80 ms、P95 不高于 160 ms；route 错误、phone clipping、跨语言 phone overlap、错误 COMPLETE 均为 0。上述数值是第一版工程验收线，不是已测模型指标。未通过时可发布纯日／纯英能力，但 `mixed.enabled` 保持 false。

## 要求到验收追踪矩阵

| 要求 | 客观验收标准 | 验证步骤 |
|---|---|---|
| R1 | 源 WAV／台本 hash 不变；每个输出含五层文本及来源 | `test_ja_en_schema.py`；运行前后 `sha256sum`；independent verifier |
| R2 | mixed 对同一文本产生 Japanese/English 两个完整 pass；exact char coverage、对应 route 选 pass、80 ms conflict 均可复现；anchor 不出现在 phone provenance | `test_ja_en_anchor.py` dual-pass／缺字／倒序／分歧边界矩阵；检查 plan/ledger source 枚举 |
| R3 | 纯日、纯英、mixed、`AI/USB`、借词和未知 token 得到预期 route／unresolved | `test_ja_en_frontend.py` 参数矩阵；真实 canary 人工标注对比 |
| R4 | 每个 runnable unit 有唯一纯 ASCII alias 且在 locked dict 恰一行；相同 surface 不同 reading 可同时对齐并反查各自 unit；未知项无 MFA phones | `test_ja_en_locked_dictionary.py` duplicate-surface fixture；verifier 重算 alias map／entry digest |
| R5 | receipt 精确记录两模型 v3.0.0、runtime 和实际 hash；size 不符即失败 | `download_models.py --include ja-en --check`；模型资产篡改负例 |
| R6 | 任意越界 phone 在启动 MFA 前返回非零；日英 namespace 分离 | inventory mutation fixture；检查输出 `ja:`／`en:` |
| R7 | 每个 local interval 往返 global sample 无偏差；clip 与 transform hash 可追溯 | `test_ja_en_audio_axis.py`；1 sample 边界 fixture |
| R8 | Japanese corpus 不含 `route=en`；token/mora/phone 顺序与 locked dict 一致 | `test_align_japanese_mfa.py`；Japanese real smoke |
| R9 | English corpus 不含 `route=ja`；短词明确 verified 或稳定拒绝原因 | 现有 English strict suite + `test_ja_en_english_bridge.py`；English real smoke |
| R10 | 越界、overlap、触边 phone 不被裁剪；日志显示双侧 retry 或 reject | `test_ja_en_seams.py`；无静音 mixed canary；比较 raw/merged phone samples |
| R11 | words/phones/language tier 单调同轴；mora 仅以 `phone_ids` 关联 | `test_merge_ja_en_mfa.py`；independent verifier |
| R12 | 任一输入／模型／词典／mapping／配置／代码摘要变化使旧 cache 失效 | `test_ja_en_resume.py` mutation matrix |
| R13 | expected = verified ∪ rejected ∪ unresolved 且两两不交；PARTIAL 返回非零或按显式 allow-partial 语义返回 | `test_ja_en_accounting.py`；manifest audit |
| R14 | 既有 MFA/Qwen/English tests 全部通过且默认下载清单行为不变 | 下节 regression 命令 |
| R15 | README 能独立说明模型、mora、例子、CLI、恢复和错误处理 | 文档 review checklist；示例配置 schema check |
| R16 | accent/F0 字段默认为 unknown/masked；无算法写伪 H/L | schema fixture；搜索输出字段来源 |
| R17 | held-out mixed 接受至少 34/40、每主桶至少 8/10、seam MAE≤80 ms、P95≤160 ms，且 route error／clipping／cross-language overlap／错误 COMPLETE 均为 0 | rollout gate script 重算人工 gold 指标；检查 publish receipt |

## 验证命令与预期信号

### 无 GPU 实施验证

以下文件／命令在实现后存在：

```bash
cd /mnt/local_E/MFA_Pause/repo
python -m compileall -q scripts tests
python -m pytest -q \
  tests/test_ja_en_schema.py \
  tests/test_ja_en_anchor.py \
  tests/test_ja_en_frontend.py \
  tests/test_ja_en_locked_dictionary.py \
  tests/test_ja_en_audio_axis.py \
  tests/test_align_japanese_mfa.py \
  tests/test_ja_en_english_bridge.py \
  tests/test_ja_en_seams.py \
  tests/test_merge_ja_en_mfa.py \
  tests/test_ja_en_resume.py \
  tests/test_ja_en_accounting.py
python scripts/run_ja_en_pipeline.py \
  --config configs/japanese_english_mfa.yaml --check
git diff --check
```

预期：全部返回 0；负例在测试内部被准确拒绝；`--check` 不创建 workspace、不加载 GPU 权重；无 inventory 外 phone、无未记账 artifact。

### 现有路径回归

```bash
cd /mnt/local_E/MFA_Pause/repo
python -m pytest -q \
  tests/test_align_english_mfa_canonical_units.py \
  tests/test_mfa_runtime_capabilities.py \
  tests/test_mfa_sharded_axis_links.py \
  tests/test_run_pipeline_mfa_root.py \
  tests/test_qwen3asr_mode.py \
  tests/test_qwen3_prealign.py \
  tests/test_qwen3_hf_backend.py
```

预期：全部通过；中文默认模型、step order、`anchored_nvv` Chinese-only 契约和 strict English v2 schema 未改变。

### 模型与真实 smoke

```bash
cd /mnt/local_E/MFA_Pause/repo
python scripts/download_models.py --only mfa --include ja-en --check
python scripts/run_ja_en_pipeline.py \
  --config /data/canary/ja_en_smoke.yaml --stage inventory,asr_anchor,frontend
python scripts/run_ja_en_pipeline.py \
  --config /data/canary/ja_en_smoke.yaml --stage align,merge,verify
python scripts/verify_ja_en_alignment.py --workspace /data/canary/run-001
```

预期信号：

- asset receipt 显示 Japanese/English acoustic + dictionary v3.0.0、实际 byte size/SHA-256、MFA runtime；
- 纯日语全部由 `ja` ledger 提供 phones，纯英语全部由 `en` ledger 提供 phones；
- mixed 样本的每个 phone 只属于一个 language run，raw phone 未被裁短；
- 无静音 seam 若无法安全合并，得到稳定 rejection，而不是伪造成功；
- `AI/USB` 日语读法和真英语样本按人工期望分路；未知 token 保持 unresolved；
- 冻结 held-out 的 mixed 自动接受至少 34/40，日→英、英→日、无停顿、短英文四桶各至少 8/10；accepted seam MAE 不高于 80 ms、P95 不高于 160 ms；
- route 错误、raw phone 裁短、跨语言 phone overlap 和错误 COMPLETE 计数均为 0；
- verifier 返回 0 后才有 `COMPLETE`，否则为 `PARTIAL` 且 mixed 发布关闭。

## 可观察验收标准

- 用户能从一份配置运行纯日、纯英、mixed 三类输入，并在 manifest 中看到逐 stem 成功／失败／未决原因。
- 每个最终 phone 可追到源 WAV、全局 sample、原始 MFA TextGrid、模型版本、词典唯一行和 unit ID。
- Japanese MFA 的任何结果都不覆盖 `route=en` unit；English MFA 的任何结果都不覆盖 `route=ja` unit。
- 同一 surface 的多读音不会以多条词典候选交给 MFA 后再把其选择称为独立识别证据。
- `学校` 一类词保留 `ガ｜ッ｜コ｜ー` 的 mora 序列及 phone ID 关系，不按四等分产生时间。
- 语言切换处没有 phone overlap／倒置／超出 WAV；出现疑似吞邻语音时能看到 retry history 或拒绝记录。
- mixed 成功不能由“全部拒收”达成：held-out 接受率、四类切换覆盖和人工 seam 误差同时达到 R17 门槛。
- 重新运行完全相同 identity 可在加载模型前恢复；改变任何模型文件、词典、override、前端 mapping 或输入音频会拒绝旧 cache。
- 现有中文和 strict English 回归全部通过，用户已有 dirty 文件内容未被覆盖。

## 风险、缓解和回滚

| 风险 | 后果 | 缓解 | 回滚 |
|---|---|---|---|
| Qwen mixed 锚点不稳定 | seam 错位，单语模型吞邻语音 | representative canary、字／词双聚合、双侧 retry、fail closed | 关闭 mixed publish，仅保留纯日／纯英；不改旧中文入口 |
| 同一汉字实际读音不同 | locked dict 锁错后 MFA 仍可能给出“看似合理”边界 | 保存独立 reading evidence；同源 G2P 不计多票；歧义过滤／override | 删除 fresh run workspace 后修 override，以新 identity 重跑 |
| kana→MFA mapping 错或新增 phone | MFA OOV／错误 phone | 配套词典 golden cases、inventory preflight、独立 verifier | 回退 mapping 版本，全部依赖 cache 自动失效 |
| legacy MFA 与 v3.0.0 资产实际不兼容 | runtime 启动或 TextGrid 格式失败 | 模型 smoke 是发布硬门 | 保持当前仓库环境不动；在隔离 env 评测兼容组合 |
| HF v3.3.0 被误当 legacy 包 | 命令和 inventory 漂移 | asset schema 明确 distribution=`legacy-v3.0.0`；HF 路线独立配置／命令 | 禁用 HF profile，继续固定 v3.0.0 |
| 短英语词没有足够上下文 | MFA 触边或被拒绝 | sample padding、与邻 run 联合重试、明确最短尝试门槛 | 保留 rejected ledger，不生成 phone |
| 能量算法把促音／清化误判为静音 | 人为切坏 phone | 能量只做诊断／重锚，不改 raw MFA phone | 丢弃 boundary candidate，恢复 raw ledger |
| 大范围改动 `postprocess_textgrids.py` 引发中文回归 | 现有生产线受损 | 日英使用新 merger；旧文件只做最小共享接口 | revert 日英调用点即可，旧 pipeline 无配置变化 |

回滚原则：所有产物位于 fresh workspace，模型与词典 run-local，旧入口默认行为不变。代码回滚只移除新入口／新模块和下载器的可选资产组；不要删除历史运行目录，也不要修改用户当前 dirty 文件。

## 未决门禁：证据、影响、责任人和决策路径

| 门禁 | 现有证据 | 影响 | 责任人 | 决策路径 |
|---|---|---|---|---|
| G1：Qwen 同句日英 anchor 质量 | 官方只声明 ja/en 支持，未承诺 code-switch；仓库现有 profile 只有中文 anchored 路径 | 决定 mixed 是否能发布 | ASR/QA | 冻结开发参数后跑 60 条 held-out；mixed ≥34/40、每桶 ≥8/10、seam MAE≤80 ms、P95≤160 ms 且四类严重错误为 0 才启用；否则保持 mixed disabled |
| G2：Latin token 业务读法 | `AI/USB` 可按日语或英语读，纯脚本无法判定 | 决定 route 和 locked dict | 数据 owner | 提供 override lexicon 或接受 unresolved；不得由实现者全局猜测 |
| G3：Japanese v3.0.0 本地 digest/runtime | 上游有固定 URL/size，无官方 digest；仓库 MFA=3.3.9 | 决定模型启动与可复现性 | runtime owner | 下载、核 size、算 SHA-256、最小 validate/align；证据写 receipt 后解除 |
| G4：kana→phone mapping | 当前仓库无实现；另一项目描述不能当本仓库证据 | 决定全部日语 pronunciation 正确性 | frontend/linguistic QA | 从配套 dict/inventory 建映射，golden cases + inventory mutation +人工抽样全过后解除 |
| G5：pyopenjtalk-plus 安装标识/API | `char_span` 依赖 v0.4.1-post9，当前 requirements 未包含 | 决定稳定 token span | environment owner | 在目标平台锁定可安装版本，跑 capability smoke 并冻结 distribution metadata |

G1-G5 不阻塞 schema、状态机、asset downloader、测试 fixture 和独立 verifier 的实现。G1 阻塞 mixed 发布；G3-G5 阻塞真实 Japanese MFA 发布；G2 仅阻塞对应歧义 token，不能扩大成静默默认。

## 执行检查清单

- [ ] 在 `/mnt/local_E/MFA_Pause/repo` 重新读取本交接和 `CLAUDE.md`，记录 HEAD／status，避开列出的用户改动。
- [ ] 创建 fresh feature branch／worktree 与 fresh run workspace；不复用任何旧对齐目录。
- [ ] 先完成 schema、error taxonomy、集合守恒和独立 verifier fixture。
- [ ] 固定 Japanese/English v3.0.0 资产并生成真实 asset receipt。
- [ ] 通过 MFA 3.3.9 capability smoke 后才实现真实 runner 调用。
- [ ] 固定 pyopenjtalk-plus API，完成文本分层和 unresolved 语义。
- [ ] 完成所有 unit 的纯 ASCII alias、单发音锁定、alias 反向映射和 phone inventory preflight。
- [ ] 完成整数 sample 轴、双模型 ledger、seam retry／reject，不实现 phone clipping。
- [ ] 完成 cache identity、atomic checkpoint、namespace 审计和 fresh publish。
- [ ] 运行无 GPU专项测试与现有回归，保存命令、版本和结果。
- [ ] 冻结开发参数后跑 60 条真实 held-out，保存 route／reading／word／seam gold 与 R17 指标报告。
- [ ] 独立 verifier 通过后才写 COMPLETE；任一门禁未过保持 PARTIAL／NO-GO。
- [ ] 若实施中修复了既有逻辑冲突 bug，按 `CLAUDE.md:1-24` 补 `REGRESSION_ARCHIVE.md`；纯新增功能不需要追加 case。

## 就绪判断

**实现规划：GO。真实 mixed 发布：有条件 NO-GO。**

仓库证据足以开始 schema、前端、双模型分段、provenance、测试和 verifier 的实现。模型选择已确定为 Japanese MFA v3.0.0 + dictionary v3.0.0，英语继续 English US ARPA v3.0.0；这与“日语单语模型不能原生覆盖英语 phones”的事实一致。真实 mixed 发布仍需 G1-G5 中相应 canary／runtime／mapping 门禁通过，这些门禁都有明确 owner、证据和解除方法，不是模糊占位。

## 下一执行窗口启动说明

1. 进入仓库：`cd /mnt/local_E/MFA_Pause/repo`。
2. 阅读本文件：`handoffs/20260921T035056Z-japanese-english-asr-mfa-pipeline.md`。
3. 执行 `git rev-parse HEAD && git branch --show-current && git status --short`，确认并保留元数据列出的用户改动；若 HEAD 或任何引用文件已变化，先刷新证据与行号。
4. 从“实施计划 0”开始，先建 schema／fixture／verifier，再碰模型调用。不要从下载大模型或修改中文 `FULL_STEP_ORDER` 开始。
5. 第一份可审查提交应只包含 schema、配置校验、状态机、fixture 和 verifier 骨架；第二份再加入模型资产／frontend；第三份加入双 MFA／merge。每份都运行对应追踪矩阵命令。
6. 在 G1 mixed canary 通过前，默认配置必须保持 `require_all_resolved: true` 且 mixed publish 关闭；纯日／纯英成功不能自动放宽 mixed 门禁。
