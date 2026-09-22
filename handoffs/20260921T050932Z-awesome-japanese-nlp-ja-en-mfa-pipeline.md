# 围绕 awesome-japanese-nlp-resources 的日英 ASR + MFA + TTS 数据管线实施交接

## 元数据

- 生成时间（UTC）：2026-09-21T05:09:32Z
- 仓库：`/mnt/local_E/MFA_Pause/repo`
- Git revision：`ff2e2f34088e61ba45f08d29d32e6f2dd3c72a44`
- 分支：`codex/0915all-full-corpus`
- 任务 slug：`awesome-japanese-nlp-ja-en-mfa-pipeline`
- 本交接：`handoffs/20260921T050932Z-awesome-japanese-nlp-ja-en-mfa-pipeline.md`
- 前版交接：`handoffs/20260921T035056Z-japanese-english-asr-mfa-pipeline.md`，保留不改；本文件是可独立执行的完整新版本。
- 规划路由：root 已显式派发 `gpt-5.6-sol`、`reasoning_effort=high`；本窗口只规划，不实现。
- 检查时已有用户／其他任务改动：`scripts/qwen3_timestamp_normalization.py`、`tests/test_reference_macro_resolutions.py` 为已修改；`configs/qwen3asr_macro_probe.yaml`、`scripts/retire_macro_source_texts.py` 和前版交接为未跟踪文件。本方案不得覆盖、清理、提交或改写它们。

## 目标与最终交付

在当前仓库实现一条从游戏原始日语音频和台本出发的完整生产管线：保守音频准备、多家族 ASR 盲听校验、日语 reading 选择、日语／英语／句内混合 MFA 音素对齐、严格证据合并，最终输出可用于 TTS 训练的音频引用、phones、durations、mora 关系和可选韵律字段。

[`awesome-japanese-nlp-resources`](https://github.com/taishi-i/awesome-japanese-nlp-resources/tree/4a1c437535f2115fdb8bdcdbc6aa2c3a60d758fd) 是固定 commit 的资源目录和选型入口，不是 Python runtime、G2P 实现、声学模型或 git submodule。项目从目录中选择并分别审计以下组件：

- `pyopenjtalk-plus`：候选主日语文本前端，用于形态素、reading、文本 phone 和字符跨度映射。
- `julius4seg` 与 JATTS 的 Julius 准备脚本：可选诊断链，用另一套表示和 forced alignment 做对照。
- Montreal Forced Aligner：唯一生产 phone timing 后端。Japanese MFA 与 English US ARPA 分别处理对应语言，绝不拼接成一个伪双语 acoustic model。

第一版完成后，每个最终 phone 必须能追到源 WAV 的全局 sample、ASR／台本 reading 决策、逐 occurrence alias、唯一词典行、MFA 模型和原始 TextGrid。任何 optional Julius 结果只写诊断 artifact，不回写 MFA 时间，不升级为读音真值。

## 用户友好的日语概念说明

### mora（拍）与 phone 不是同一个东西

`さくら`（sakura，樱花）分为 `さ｜く｜ら`，是 3 拍。每拍可对应多个辅音／元音 phone，因此不能把 3 拍理解成 3 个 phone。

`東京`（Tōkyō，东京）按读音可写作 `と｜う｜きょ｜う`，即 `to｜o｜kyo｜o`，是 4 拍。长元音经不同工具会采用不同表示：

- `julius4seg/converter.py` 的 `conv2julius` 规则把 normalized hiragana `とーきょー` 表为 `t o: ky o:`。
- 同文件的 `conv2openjtalk` 规则把它展开为 `t o o ky o o`。
- 已核验的官方 Japanese MFA dictionary v3.0.0 中，`東京` 的条目为 `t oː c oː`。

这三行只说明**表示体系不同**。前两行是固定源码转换规则的推导，不是声称某个本机 OpenJTalk 版本已经对 `東京` 做过实测。MFA 条目中的 `c` 是该日语模型 inventory 的标签，不能因为字形相同就当成拼音 `c`，也不能用字符串替换成 `ky`。

长元音的一个 `oː` phone 可以关联 `o｜o` 两拍，所以 mora 与 phone 是多对多关系。把数组按索引 `zip`，或把 `oː` 的时长一分为二，都会制造没有声学证据的内部边界。默认只保存 `mora_ids <-> phone_ids` 关系；若未来有声学 mora estimator，其时间必须标 `estimated`、记录模型和置信度，并从训练 target mask 掉，不能冒充 MFA 真值。

促音（小 `っ`）的闭塞和清化／无声化元音也不是普通静音。能量低不等于可删除；边界算法不得把它们归为 silence 后吞掉。日语文本重音预测同理：pyopenjtalk-plus 或 Marine 给出的 accent 是文本预测，真实 F0 是音频测量，两者必须使用不同字段、provenance 和 mask。

## 当前仓库事实

1. 主配置仍以 `mandarin_mfa`、`dict/mfa_ipa.dict` 和拼音词典为中文默认；英语另有 `mfa_en` 配置，使用 `english_us_arpa`、CMUdict 和英语 G2P（`scripts/run_pipeline.py:577-595`、`scripts/run_pipeline.py:665-692`）。
2. 当前中英混合已经采用“从全局 CTC 轴提取英语 runs、单独英语 MFA、strict ledger 回填”的多模型结构，而非一个双语 acoustic model：英语分段见 `scripts/align_english_mfa.py:977-1098`，run-local 词典见 `scripts/align_english_mfa.py:1314-1440`，隔离 MFA 调用见 `scripts/align_english_mfa.py:1520-1563`。
3. Qwen prealign 明确只提供 lexical spans，MFA 保持 phone aligner（`scripts/qwen3_prealign.py:1-7`）。Qwen transcript config 可把 auto 归一到 `language=None`，但现有 `anchored_nvv` profile 强制中文，不能直接复用成日英 profile（`scripts/qwen3asr_transcribe.py:191-247`）。
4. 当前默认步骤顺序把中文 normalize、MFA、English MFA 和中文后处理连在一起（`scripts/run_pipeline.py:6955-6975`）。日英前端和 phone 语义不同，应新增独立入口，避免修改中文默认顺序。
5. 英语 phone 使用 ARPABET inventory，并已有 `en:` 输出前缀（`scripts/pipeline_utils.py:1350-1352`、`scripts/pipeline_utils.py:2178-2215`）。日语 adapter 不得调用现有 IPA→ARPABET 或 IPA→拼音表。
6. 当前环境固定 Montreal Forced Aligner `3.3.9`（`requirements.txt:9-37`、`environment.yml:6-35`）。模型下载器已有中文、英语、Qwen 资产，但没有日语资产（`scripts/download_models.py:53-69`、`scripts/download_models.py:141-169`）。
7. 当前英文 OOV cache 只以 OOV 词表摘要命名（`scripts/align_english_mfa.py:1408-1421`）。新前端 cache 必须额外绑定源码 commit、API flags、mapping、模型、字典和输入，不得延续这个较弱签名。
8. `CLAUDE.md:1-24` 要求：若实施过程中修复已有逻辑冲突 bug，需要补 `REGRESSION_ARCHIVE.md`；本任务中的纯新增功能不触发该要求。

## 上游资源事实与固定来源

### 资源选型表

| 资源 | 固定 revision | 已核实用途 | 许可与供应链处理 | 本项目地位 |
|---|---|---|---|---|
| awesome-japanese-nlp-resources | `4a1c437535f2115fdb8bdcdbc6aa2c3a60d758fd` | 日语 NLP 资源目录 | 仓库为 CC0-1.0；只记录 commit、README hash 和选型来源，不作为依赖安装 | 发现／审计索引 |
| pyopenjtalk-plus | `9e4bf25324ac135dfc81ca64aed2fa6a48b83304` | `g2p_mapping`、`run_frontend_detailed`、`make_phoneme_mapping`、reading／accent／POS／char span | 仓库 API 的 license 字段为 NOASSERTION，但固定 commit 的 `LICENSE.md` 开头是 MIT Expat；包源码、Open JTalk、MeCab 字典、tsqyomi／Marine 模型仍分别审计和 pin | 候选主日语前端 |
| julius4seg | `e14beae2940fd5a6ac5a9d2afc249eac6fac4a50` | `conv2julius`／`conv2openjtalk` 表示转换，Julius segmentation 工具 | 代码 MIT；Julius binary、声学模型、字典和数据许可另记 | 可选诊断 backend |
| JATTS | `a5a8cd0b9a92caa065b1b6cc4cf8e805e56e4c61` | 日语文本转 hiragana、16 kHz PCM16 准备、Julius `-palign` 和 lab→durations 的参考流程 | 代码 MIT；调用的外部 binary／模型／语料许可另记 | 诊断编排参考，不整仓 vendoring |
| Japanese MFA acoustic + dictionary | v3.0.0 legacy assets | 日语 production phone alignment | 模型和字典各自记录 URL、byte size、本地 SHA-256、metadata、license；不能用代码仓库许可替代模型许可 | 必需生产后端 |
| English US ARPA acoustic + dictionary | v3.0.0 legacy assets | 真英语 production phone alignment | 同上，独立 inventory 和 receipt | 必需生产后端 |

Japanese MFA v3.0.0 的官方资产定位固定为：acoustic archive
`https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-japanese_mfa-v3.0.0/japanese_mfa.zip`
（92,191,596 bytes）和 dictionary
`https://github.com/MontrealCorpusTools/mfa-models/releases/download/dictionary-japanese_mfa-v3.0.0/japanese_mfa.dict`
（21,264,022 bytes）。这些 byte size 只辅助识别下载对象，不是完整性证明；实施时必须计算并冻结实际下载文件的 SHA-256，再由 receipt 验证。

固定源码证据：

- [pyopenjtalk-plus README at pinned commit](https://github.com/tsukumijima/pyopenjtalk-plus/tree/9e4bf25324ac135dfc81ca64aed2fa6a48b83304)
- [pyopenjtalk-plus mapping types](https://github.com/tsukumijima/pyopenjtalk-plus/blob/9e4bf25324ac135dfc81ca64aed2fa6a48b83304/pyopenjtalk/types.py)
- [julius4seg converter.py](https://github.com/Hiroshiba/julius4seg/blob/e14beae2940fd5a6ac5a9d2afc249eac6fac4a50/julius4seg/converter.py)
- [JATTS prepare_julius.py](https://github.com/unilight/jatts/blob/a5a8cd0b9a92caa065b1b6cc4cf8e805e56e4c61/utils/prepare_julius.py)
- [JATTS segment_julius.pl](https://github.com/unilight/jatts/blob/a5a8cd0b9a92caa065b1b6cc4cf8e805e56e4c61/utils/segment_julius.pl)
- [JATTS data_prep_post_julius.py](https://github.com/unilight/jatts/blob/a5a8cd0b9a92caa065b1b6cc4cf8e805e56e4c61/utils/data_prep_post_julius.py)

### pyopenjtalk-plus API 事实与陷阱

- README 记录 v0.4.1-post8 加入 `g2p_mapping`、`run_frontend_detailed`、`make_phoneme_mapping`，post9 加入输入字符的 `char_span`。本方案 pin commit 和 capability，不从 README 下半部复制原版 `pip install pyopenjtalk` 示例。
- 该包兼容 `pyopenjtalk` import namespace。未经验证同时安装 `pyopenjtalk` 和 `pyopenjtalk-plus` 会产生 distribution/module ownership 歧义，因此 production frontend 环境只能安装一个候选，并用 `importlib.metadata.packages_distributions()`、版本、源码 commit 和 wheel SHA-256 证明实际提供者。
- `g2p_mapping(text=caller_text)` 的 `SurfacePhonemeMapping.char_span` 指向 caller text 的半开区间。直接调用 `make_phoneme_mapping` 时：提供 `morphs + caller_text` 才映射 caller；只提供 morphs 时映射 MeCab normalized text；都不提供时映射 NJD surface 拼接文本。后两种结果不能冒充 `orig_text` span。
- `MeCabMorph.char_span` 指向 `text2mecab` 规范化文本。无法对应 morph entry 的 mapping 会得到 `(0, 0)`；任何非空 lexical unit 的 `(0, 0)` 必须 fail closed，不能解释成真实开头位置。
- `SurfacePhonemeMapping.is_ignored` 表示该 surface 的 phone 列表为空，例如开头长音符；它不是 MeCab 层“空白／标点”判定。producer 必须分别保存 `morph_ignored`、`phoneme_empty` 和 `lexical_status`。
- `use_tsqyomi` 是可选上下文 reading 模型；它仍是文本推断，不是音频证据。`use_read_as_pron`、`revert_long_vowels`、`revert_yotsugana` 会改变 reading／phone 表示；`run_marine` 可改变重音预测。所有开关必须在配置中显式布尔化并进入 cache identity，禁止依赖包默认值。

### Julius4seg/JATTS 事实边界

- `converter.py` 提供表示转换规则，不证明实际录音或某个 OpenJTalk build 的输出。
- JATTS `prepare_julius.py` 参考流程用 `pyopenjtalk.g2p(..., kana=True)`，再由 jaconv 转 hiragana，并准备 16 kHz PCM16 音频。
- `segment_julius.pl` 用 Julius `-palign` 做 forced alignment；`data_prep_post_julius.py` 读取 `.lab` 的 start/end/phone 并计算 durations。
- 这些能力适合做第二实现的诊断对照，但没有任何一项允许 Julius 覆盖 MFA production timestamps，或把两个 forced aligner 的一致称为独立 ASR reading 投票。

## Facts、Assumptions、Decisions、Open Questions

### Facts

- 资源目录只说明“有哪些项目”，不保证 API、模型、许可、兼容性或质量；每个选中组件需独立 pin 和审计。
- pyopenjtalk-plus 是文本前端。它能给 reading、文本 phone、accent 和 span，不能从音频求真实 phone timing。
- 同一汉字台本与 ASR 输出相同，不证明录音采用前端给出的读音；多家 ASR 最终若都走同一个 G2P，也共享同一个推导误差。
- MFA 根据给定 transcript/dictionary 对齐，MFA score、phone sequence 和 Julius 对齐都不是新的 ASR 读音票。
- Japanese MFA 与 English US ARPA 的 phone inventories 不同。Japanese 模型不能通过添加英语 phone 变成双语模型，两个 native label 也不能因字符串相同而自动视为同义。

### Assumptions

- 游戏数据能整理为稳定 `uid`、源 WAV、台本、游戏／说话人和可选人工 override；具体 GS/SR/WW 路径由 manifest 配置，不硬编码到脚本。
- 至少能为 baseline 三个 ASR 家族准备合法本地模型和运行环境；缺少家族时只允许开发 smoke，不发布高置信 reading。
- 数据 owner 接受歧义读音、无法绑定 span 和 unsafe seam 被过滤，而不是强行生成 TTS target。
- Japanese MFA v3.0.0 与当前 MFA 3.3.9 的真实兼容性尚未由本方案验证，必须通过 runtime gate。

### Decisions

1. 新增独立 `run_ja_en_pipeline.py`；不把日英 frontend 混入当前中文 `FULL_STEP_ORDER`。
2. awesome 目录不进入 runtime；新增机器可读 supply-chain lock，记录所选源码、wheel、binary、模型、字典和许可证据。
3. pyopenjtalk-plus 固定 commit 是**候选主前端**。通过 capability、span、表示和许可门禁后才升级为 production frontend；失败时状态为 blocked，不静默换回另一个 `pyopenjtalk` distribution。
4. ASR production baseline 是三个家族：Qwen3-ASR、Whisper large-v3、ReazonSpeech NeMo。extended profile 增加 kotoba-v2 和 ReazonSpeech k2，但同家族多个模型最多贡献一个 family vote。
5. Qwen3-ForcedAligner 负责全句候选 lexical anchors；MFA 负责最终 phones。混合语句沿用 Japanese／English 双 pass、按字符跨度聚合和 seam 双侧重跑／拒收。
6. 日语生产模型固定 Japanese MFA acoustic v3.0.0 + dictionary v3.0.0；英语固定 English US ARPA v3.0.0 + 同版字典。HF 新模型是独立升级实验。
7. 每次 token occurrence 生成唯一 ASCII alias。基础字典内词、多音词和 OOV 一律把本次已选发音锁成 alias 的唯一词典行，不让 MFA 在多 pronunciation 中代替 reading selector 决策。
8. OpenJTalk／Julius／MFA 之间使用 semantic phone graph 和 target-model adapter；禁止直接字符串 replace、直接索引 `zip` 或跨模型 inventory 合并。
9. Julius4seg/JATTS 是 optional diagnostic backend。它写自己的时间、转换和差异报告，不进入 reading vote，不改 MFA raw／merged timestamps。
10. TTS JSONL 同时保留 model-native phones、semantic relationships、mora mapping 和证据 mask；accent prediction 与 measured F0 分层。

### Open Questions 摘要

- pinned pyopenjtalk-plus commit 在目标平台能否构建可复现 wheel，并满足所有 API／span canary。
- 三个 ASR baseline 的精确模型 revision、runtime、模型许可和 GPU 预算需要由运行 owner 冻结。
- Japanese MFA v3.0.0 的实际 inventory、资产 SHA-256、许可和 MFA 3.3.9 runtime smoke 尚需产出 receipt。
- semantic adapter 对全部 Japanese MFA phones 的覆盖与语言学审查尚未完成；`東京` 一个条目不能证明全表可映射。
- Julius binary／声学模型／词典的合法固定资产尚未选定，因此 diagnostic backend 默认关闭。
- 数据集中 `AI`、`USB`、日语化借词与真英语的 policy lexicon 需数据 owner 提供。

## 范围、非范围与硬约束

### 本次实现范围

- 原始音频 inventory、保守处理、训练 WAV 与 16 kHz alignment WAV 的时间轴收据。
- baseline／extended ASR provider、家族去相关计票、reading 选择和未决账本。
- pyopenjtalk-plus frontend contract、span 校验、semantic graph、MFA target adapter、逐 occurrence alias 词典。
- 日语／英语分段 MFA、mixed seam、strict merge、Julius optional diagnostic、TTS JSONL／TextGrid。
- schema、cache、恢复、独立 verifier、供应链 lock、真实 canary 和现有中文／英语回归。

### 非范围

- 把 awesome 目录整体安装、整仓 vendoring 或让它决定运行版本。
- Julius-only production、用 Julius 替代 MFA、或将 Julius timestamps 写回 production output。
- 从同一 G2P 派生的相同 reading 冒充多个独立声学证据。
- 训练新的 ASR／MFA／accent／F0 模型，或把文本重音预测标成真实基频。
- 根据低能量自动删除促音闭塞、清化元音或其它日语弱音段。
- 把长元音单 phone 均分成两个已知 mora 边界。

### 不变量

- 原始 WAV、台本、旧产物和本窗口已有改动只读；所有执行写 fresh workspace。
- 整数 sample 是时间权威；train/alignment audio 之间有可重算 transform receipt。
- 每个 stem、unit、language run 精确分入 verified、rejected、unresolved；集合守恒。
- 所有 corpus aliases 唯一且可反查 occurrence；每个 alias 在 run-local dict 恰一条 pronunciation。
- 每个 target phone 都属于对应 acoustic model inventory；日英 inventory 分开校验。
- MFA／Julius 的 score 或输出不改变 reading evidence 的 family vote。
- raw frontend、raw MFA、raw Julius artifacts 永不原地覆盖；任何派生修正保留 provenance。

## 目标架构和阶段产物

```text
game raw WAV + origin script + speaker metadata
  -> inventory / conservative audio transform / immutable sample axis
  -> ASR blind suite
       Qwen family + Whisper family + Reazon family
  -> reading evidence and family-aware selection
  -> pyopenjtalk-plus frontend contract
       caller spans + morph/readings + text phones + accent prediction
  -> semantic phone graph + per-occurrence aliases
  -> Qwen ForcedAligner lexical anchors and language routing
  -> language runs
       ja -> Japanese MFA v3.0.0
       en -> English US ARPA v3.0.0
  -> strict sample-axis merge / seam retry or reject
  -> optional Julius diagnostic branch
  -> independent verifier
  -> TextGrid + TTS JSONL + receipts + rejection ledger
```

### Stage 0：供应链和输入冻结

输入：source manifest、固定资源 revision、模型路径。输出：`supply_chain_lock.json`（schema `ja-supply-chain-lock-v1`）、`source_inventory.json`、音频 header/hash、台本 hash。lock 对每项记录 source URL、commit/tag、tree digest、构建命令、wheel/binary/model hash、license file hash、模型／字典许可状态和审核人；未知许可状态不允许 production。

### Stage 1：音频准备

保留源 WAV；生成训练 WAV 和 16 kHz mono PCM alignment WAV。内部静音处理默认关闭；边缘 trim/pad 只能由明确阈值执行并记录 sample transform。促音／清化元音保护只在词内生效，不将低能量视为删除授权。输出 `audio_transform_receipt-v2`，包含 source/output frame count、sample rate、channel policy、hash、offset 和 duration。

### Stage 2：ASR 盲听与 reading evidence

ASR profiles：

| profile | providers | 生产语义 |
|---|---|---|
| `qwen_only_dev` | Qwen3-ASR-1.7B | 只做开发、transcript 和 anchor smoke；reading confidence 不得升级为 production verified |
| `baseline_3family` | Qwen3-ASR-1.7B、Whisper large-v3、ReazonSpeech NeMo v2 | production 最小基线；每家族最多一票 |
| `extended_5model` | baseline + kotoba-v2 + ReazonSpeech k2 v2 | 观察家族内稳定性；kotoba 不给 Whisper 增票，k2 不给 Reazon 增票 |

每个 provider 原样保存 `asr_text`、模型 revision/tree hash、family、runtime 和失败。NFKC 本身有损；共同 normalizer 必须保留 `orig_text`／raw ASR text，并为 NFKC、标点、Ruby、placeholder 和 kana 比较产物保存显式 text-layer offset/provenance 映射，使结果可回溯到原始字符，不能声称规范化操作可逆。每个候选 reading 标：`surface_source`、`frontend_id`、`family`、`is_same_surface_g2p`、`evidence_scope`。

reading selector 固定顺序：

1. 有人工 `reading_override` 且 `source_id` 完整：`manual_verified`。
2. 任一 ASR family 对台本达到 exact surface 或 kana match：选择台本 contextual reading，状态 `origin_surface_confirmed`；明确标记这是 lexical support，不声称 ASR 听出了同汉字内部读音。
3. 至少两个不同 ASR families 产生相同 kana candidate：`asr_family_consensus`。若候选都来自相同汉字经同一 frontend，附 `shared_g2p_derivation=true`，置信度不得等同人工读音。
4. 没有跨家族一致时可计算 `asr_medoid` 供诊断，但默认不进入 production MFA。
5. 其余为 `none/unresolved`。

Qwen ForcedAligner、MFA 和 Julius 均不加入上述票数。

### Stage 3：pyopenjtalk-plus frontend contract

计划 schema `ja-frontend-contract-v2`。每次调用输入：

```json
{
  "caller_text": "東京でgameをする",
  "text_layer_digest": "042c4d97dac3f29c4e5de3409aaf43973725248a912782bf287a46d852376d20",
  "frontend_commit": "9e4bf25324ac135dfc81ca64aed2fa6a48b83304",
  "options": {
    "use_tsqyomi": false,
    "use_read_as_pron": false,
    "revert_long_vowels": false,
    "revert_yotsugana": false,
    "run_marine": false
  }
}
```

production 调用必须通过 `g2p_mapping(text=caller_text)`，或显式 `run_frontend_detailed(caller_text)` 后以 `make_phoneme_mapping(morphs, caller_text)` 绑定 caller；禁止只给 morphs 后把 normalized spans 当 caller spans。输出逐 unit 保存 caller `char_span`、MeCab normalized span、surface、pronunciation、reading、POS、accent phrase／nucleus、text phone list、ignored flags、API options 和 frontend tree/wheel digest。

Stage 3 先保存整句 frontend 的原始映射，再按 Stage 2 的每个 occurrence `selected_reading` 应用 override 并重建该 occurrence 的 mora／semantic phones。`selected_reading` 是锁定输入，禁止再次调用 `g2p(surface)` 用默认读音覆盖它。override 改变原始 reading 时，原 frontend accent／nucleus 标为失效并清除训练 mask，除非后续明确按锁定 reading 重新生成且保留新 provenance。进入 Stage 4 前必须校验最终 phones 可由锁定 reading 重建；校验失败进入 `dictionary_roundtrip_failed`／`reading_ambiguous`，不得退回 surface 默认读音。

错误分流固定为：

- `frontend_provider_ambiguous`：两个 distribution 都声明 `pyopenjtalk` namespace，或实际 provider 与 lock 不符。
- `frontend_capability_missing`：固定 API／字段不存在。
- `frontend_span_unbound`：非空 lexical mapping 为 `(0,0)`、越 caller text 或重叠倒置。
- `frontend_empty_lexical_phones`：lexical unit 被标 `is_ignored` 且无可解释规则。
- `frontend_representation_drift`：同 identity 重跑 reading／phones／span 不同。
- `reading_ambiguous`：同 occurrence 候选未决。

上述错误不能回退到 substring 搜索或旧 `pyopenjtalk`。

### Stage 4：semantic phone graph 与 MFA adapter

中间层不是字符串数组，而是版本化 graph：

```text
unit occurrence
  -> mora nodes: surface, kana, kind, source span
  -> semantic phone nodes: consonant/vowel/mora-nasal/geminate/length/silence features
  -> representation edges: frontend, Julius, Japanese-MFA
  -> timing edges: MFA raw interval IDs only
```

adapter 分两步：

1. `openjtalk_to_semantic`：将 frontend phones、reading 和 mora 解析成语义节点，保留长音展开、促音、拨音、清化候选，不做模型标签猜测。
2. `semantic_to_japanese_mfa_v3`：用审计过的显式表和配套 dictionary golden rows生成 Japanese MFA v3.0.0 phones；输出必须全部在实际模型 inventory 内，并能从 chosen reading 重建。`c`、`ky`、`o o`、`o:`、`oː` 等只通过语义／模型 adapter 关联。

错误分流：`semantic_parse_failed`、`semantic_relation_ambiguous`、`mfa_phone_unsupported`、`mfa_inventory_mismatch`、`dictionary_roundtrip_failed`、`mora_phone_relation_unresolved`。任何错误都使 occurrence unresolved/rejected，不允许删 phone 继续。

每个 occurrence 分配稳定 ASCII alias：日语 `ju_`、英语 `eu_` 加排序 ordinal。MFA transcript 只写 alias，`locked.dict` 每 alias 恰一行，`alias_map.jsonl` 保存 `alias -> stem/unit/surface/reading/native phones/source`。因此同一 surface 的不同 reading、基础字典多 pronunciation 和 OOV 都不会把选择权交给 MFA。

### Stage 5：Qwen anchor、语言 runs 与双 MFA

语言路由优先级：人工／项目 lexicon override > 已验证 Japanese reading > 明确 English pronunciation > ASR lexical evidence > unresolved。`AI`、`USB`、游戏专名和英语借词不能只按脚本分类。

纯日语 Qwen ForcedAligner 使用 `language=Japanese`；纯英语使用 `language=English`。mixed 对同一完整 `spoken_text` 和 WAV 固定跑两个 pass：Japanese 与 English。每个 pass 的 raw items 通过相同 lexical character stream exact monotonic 对齐到 unit char spans；Japanese units 取 Japanese pass，English units 取 English pass。每个 seam 分别计算两 pass 中左右 unit 间的整数 sample 中点；两估计相差超过 80 ms 则 `anchor_conflict`，否则取两估计的整数均值下界作为初始 ownership seam。

相邻同语言 units 合为 run，裁片保留上下文 padding，但 ownership 不扩张。Japanese runs 用 Japanese MFA v3.0.0，English runs 用 English US ARPA v3.0.0；各自 dict、inventory、MFA root、temp、log 和 strict ledger 隔离。Japanese 模型产生的 English 占位 phone、English 模型产生的 Japanese 占位 phone一律不是最终证据。

MFA 输出映回全局 sample 后，出现 phone 越 ownership、两侧 overlap、首尾 phone 贴裁片、token cardinality 不一致时，只允许扩大／平移上下文并同时重跑 seam 两侧。达到 retry 上限仍失败则拒绝 seam／utterance。禁止裁短 phone、比例缩放或用能量低点直接替换 phone 边界。

### Stage 6：optional Julius diagnostic

该 stage 默认 `enabled: false`，运行在独立环境和 namespace：

1. 读取同一 frozen 16 kHz alignment WAV、selected reading 和 unit IDs。
2. 用 pinned `conv2julius` adapter 生成 Julius 表示；保存原 reading、semantic graph、Julius phones 和转换 digest。
3. 按 JATTS 参考契约检查 16 kHz PCM16，调用 pinned Julius binary/model 的 `-palign`，解析 `.lab` 为 diagnostic durations。
4. 仅在 semantic 层比较 phone sequence、总覆盖、边界偏差、长元音／促音表现，写 `julius-diagnostic-v1`。
5. 不将 Julius reading、score 或 timing 加入 ASR votes，不回写 `strict-ja-mfa-v2`、merged TextGrid 或 TTS durations。缺 binary／model／许可时写 `diagnostic_unavailable`，不触发 production fallback。

### Stage 7：TTS 输出与韵律字段

production 输出 schema：

- `ja-en-alignment-v2`：words、native phones、global samples、language、raw MFA interval IDs、mora graph、seams、model／dict／frontend identity。
- `tts-training-record-v1`：train WAV、alignment transform、phones、durations、phone→mora／mora→phone、selected reading、speaker、text layers、quality masks。
- TextGrid：`words`、`phones`、`language` 三个必需 tier；phones 使用 `ja:`／`en:` namespace。mora 多对多关系以 JSON graph 为权威。

韵律字段分开：`accent_predicted` 来自固定文本前端／Marine并带 model digest；`f0_measured` 只能来自音频算法；`accent_known_mask`、`f0_known_mask` 独立。未知、补值、无声 phone 必须有 mask。默认不产生伪 mora internal timing；可选 estimator 输出单独 `estimated_mora_spans`，不得进入真实 duration target。

## 计划配置

以下为计划新增接口，不是当前可运行配置：

```yaml
pipeline: ja_en_tts
workspace: /data/runs/ja_en_v1
input_manifest: /data/manifests/game_ja.jsonl
supply_chain_lock: third_party/japanese_nlp_sources.lock.json

asr:
  profile: baseline_3family
  family_vote_policy: one_per_family
  qwen_model: /models/qwen3-asr
  qwen_forced_aligner: /models/qwen3-forced-aligner
  whisper_model: /models/whisper-large-v3
  reazon_nemo_model: /models/reazonspeech-nemo-v2

frontend:
  provider: pyopenjtalk-plus
  commit: 9e4bf25324ac135dfc81ca64aed2fa6a48b83304
  use_tsqyomi: false
  use_read_as_pron: false
  revert_long_vowels: false
  revert_yotsugana: false
  run_marine: false
  reject_unbound_spans: true

mfa:
  runtime_python: /envs/mfa/bin/python
  dither: 0.0
  japanese_acoustic: /models/mfa/japanese_mfa-v3.0.0.zip
  japanese_dictionary: /models/mfa/japanese_mfa-v3.0.0.dict
  english_acoustic: /models/mfa/english_us_arpa-v3.0.0.zip
  english_dictionary: /models/mfa/english_us_arpa-v3.0.0.dict

mixed:
  enabled: false
  anchor_strategy: dual_full_utterance_v1
  anchor_max_disagreement_ms: 80
  padding_ms: 100
  max_retries: 2
  allow_phone_clipping: false

julius_diagnostic:
  enabled: false
  converter_commit: e14beae2940fd5a6ac5a9d2afc249eac6fac4a50
  jatts_reference_commit: a5a8cd0b9a92caa065b1b6cc4cf8e805e56e4c61
  write_back: false

publish:
  require_supply_chain_licenses: true
  require_all_resolved: true
  require_independent_verifier: true
```

## Cache、恢复和 schema 版本

每个 stage 的 cache key 是规范 JSON SHA-256，只包含该 stage 的直接输入、实现身份和 DAG 上游依赖身份：source WAV／text hashes、所用 ASR model trees 与 family policy、Qwen anchor model、pyopenjtalk-plus commit/wheel/provider、所用 frontend flags、caller text digest、semantic adapter table digest、MFA runtime/model/dictionary/inventory、alias map、audio transform、相关配置、schema 和实现文件 hashes。Julius binary/model/converter identity 只进入 diagnostic stage 的 cache 及其 diagnostic 下游；production MFA／TTS DAG 不依赖 diagnostic artifact，因此切换 Julius 开关或模型不能改变 production cache key 或内容 hash。

计划 schema：

- `ja-supply-chain-lock-v1`
- `ja-asr-evidence-v2`
- `ja-reading-selection-v2`
- `ja-frontend-contract-v2`
- `ja-semantic-phone-graph-v1`
- `ja-en-alignment-plan-v2`
- `strict-ja-mfa-v2`
- 兼容读取 `strict-en-mfa-v2`
- `julius-diagnostic-v1`
- `ja-en-alignment-v2`
- `tts-training-record-v1`

checkpoint 先写同目录临时文件再原子替换。resume 在加载 GPU／MFA／Julius 前重算 namespace 和 hashes；identity drift、额外文件、symlink、缺 artifact 或 schema 变化均 fail closed。完整运行只在独立 verifier 通过后写 `COMPLETE`，其余为 `PARTIAL`；旧 cache 不做隐式迁移。

## 影响文件和符号

| 文件／符号 | 当前证据 | 计划变更 |
|---|---|---|
| `third_party/japanese_nlp_sources.lock.json`（新增） | 当前无日语资源供应链清单 | 记录 awesome／pyopenjtalk-plus／julius4seg／JATTS 固定 commit、许可、构建与 artifact hashes；模型和字典分项审计。 |
| `requirements-ja-frontend.txt`、`environment-ja-frontend.yml`（新增） | 当前默认依赖是中文 MFA 环境，见 `requirements.txt:1-21` | 建独立 frontend 环境，只允许一个 `pyopenjtalk` namespace provider；pin commit-built wheel和能力。 |
| `scripts/verify_ja_supply_chain.py`（新增） | 下载器只查模型存在性 | 校验 lock、license hashes、wheel/module ownership、binary/model hashes 和未知许可门禁。 |
| `scripts/run_ja_en_pipeline.py`（新增） | 中文 steps 固定于 `scripts/run_pipeline.py:6955-6975` | 编排 inventory、ASR、reading、frontend、adapter、MFA、diagnostic、merge、TTS、verify。 |
| `scripts/ja_asr_crossval.py`（新增） | 当前仓库仅有 Qwen 主线 | 实现 provider manifest、family-aware votes、origin／consensus／medoid／none 和 evidence scope。 |
| `scripts/ja_frontend.py`（新增） | 当前无日语 frontend contract | 包装 pinned pyopenjtalk-plus，强制 caller spans、flag identity、错误分流和 raw artifact。 |
| `scripts/ja_phone_adapter.py`（新增） | 当前中文／英语 phonemap 不适用于 Japanese MFA | 实现 semantic graph、OpenJT→semantic→Japanese MFA v3 adapter、inventory和dictionary roundtrip。 |
| `scripts/ja_en_schema.py`（新增） | English 有独立 canonical schema | 定义所有 v2 plan／ledger／TTS schema、集合守恒和原子 writer。 |
| `scripts/align_japanese_mfa.py`（新增） | 当前无 Japanese runner | alias corpus、locked dict、isolated MFA、raw TextGrid parser、strict-ja ledger。 |
| `scripts/align_english_mfa.py:build_en_dict/run_en_mfa` | 当前有 English OOV dict 与隔离 runner，见 `scripts/align_english_mfa.py:1314-1440`、`1520-1563` | 提取不破坏旧 CLI 的共享 runner；新入口使用 occurrence alias，不改变旧 strict-en 输出。 |
| `scripts/merge_ja_en_mfa.py`（新增） | 当前 English 注入依赖中文 postprocess | 用整数 sample 合并双 MFA、seam retry／reject、mora graph；不调用中文 phonemap。 |
| `scripts/run_julius_diagnostic.py`（新增） | 当前无 Julius | 固定 converter/JATTS contract，隔离执行和 diagnostic-only receipt。 |
| `scripts/verify_ja_en_tts.py`（新增） | 当前无对应独立 verifier | 独立重算 supply chain、span、reading、alias、inventory、sample、seam、TTS masks 和集合守恒。 |
| `scripts/download_models.py:MFA_MODELS/download_mfa` | 当前清单无 Japanese assets，见 `scripts/download_models.py:53-69` | 增加 opt-in `ja-en` 固定 v3.0.0资产组和本地 hash receipt；默认下载行为不变。 |
| `configs/japanese_english_tts.yaml`（新增） | 当前无 ja-en 配置 | 提供 fail-closed 示例，mixed/Julius 默认关闭。 |
| `README.md`、`docs/QWEN3ASR_MODE.md` | 只描述当前中文／Qwen模式 | 增加资源选型、ASR profiles、frontend contract、MFA模型、TTS输出和限制。 |
| `tests/test_ja_*.py`、fixtures（新增） | 当前已有 MFA root／English strict／Qwen回归 | 覆盖供应链、provider冲突、span、flags、adapter、alias、ASR family、MFA、mixed、Julius隔离、cache、TTS。 |

## 编号需求

**R1. 资源目录边界。** awesome 固定 commit 只作发现／审计索引，不得成为 runtime import、隐式最新版或许可代理。

**R2. 供应链。** 代码、wheel、binary、声学模型、字典和可选模型分别 pin revision/hash/license；未知许可阻止 production。

**R3. 原始数据和音频轴。** 游戏 WAV、台本、speaker、uid 可追溯；train/alignment 音频共享可逆 sample transform，不删除日语弱音现象。

**R4. ASR profiles。** 明确 qwen-only dev、三家族 production baseline 和五模型扩展；同家族不得增加独立票数。

**R5. Reading selection。** origin、family consensus、medoid、none 和 manual 状态可复现；相同汉字同 G2P 不伪称声学读音证明；MFA/Julius不投票。

**R6. Frontend 环境。** production 只允许一个 `pyopenjtalk` namespace provider，commit/wheel/API capability 与 lock 一致。

**R7. Frontend span。** caller text、MeCab normalized text、NJD surface 的坐标域显式；lexical `(0,0)`、越界、倒置、empty phones fail closed。

**R8. Frontend options。** tsqyomi、read-as-pron、long-vowel、yotsugana、Marine全部显式配置并进入 cache；文本推断不标音频证据。

**R9. Semantic adapter。** OpenJT、Julius、Japanese MFA 通过 semantic graph 和显式 model adapter连接，不用字符串 replace或数组索引配对。

**R10. 日语模型。** production 固定 Japanese MFA acoustic v3.0.0 + dictionary v3.0.0，并通过 MFA 3.3.9真实 capability、inventory、license和asset receipt。

**R11. 英语模型。** production 真英语固定 English US ARPA v3.0.0；日英 inventories／词典／MFA roots 隔离。

**R12. Occurrence alias。** 所有 IV/OOV／多音 occurrence 都用唯一 ASCII alias 和唯一 pronunciation，可反查原 token 与 reading。

**R13. Mixed alignment。** 纯日、纯英和 mixed 可验收；dual Qwen anchors、language runs、global sample mapping、seam 双侧重跑／拒绝不退化。

**R14. Mora/phone。** 保存多对多 graph、长元音／促音／拨音关系；无证据时不产生真实 mora internal timing。

**R15. Julius 诊断。** optional backend 只读同一输入、隔离产物、不参与读音投票、不回写 MFA/TTS timing，缺资产不触发 fallback。

**R16. Accent 与 F0。** 文本 accent prediction、measured F0、未知／补值／无声 mask 分层。

**R17. TTS 输出。** 每条记录含 train WAV、sample transform、native phones、durations、mora graph、reading、speaker、text layers和quality masks。

**R18. Cache／恢复。** cache identity 覆盖全部源码、模型、flags、mapping、audio、text、config和schema；篡改／extra／drift均拒绝。

**R19. 错误和集合守恒。** 每个 expected stem/unit/run精确分区，稳定错误码和证据路径；PARTIAL不得伪装COMPLETE。

**R20. 兼容性。** 现有中文步骤、Qwen中文profile、strict-en v2、MFA root隔离和模型下载默认行为保持不变。

**R21. 中文文档。** 不会日语的用户能理解 sakura、Tōkyō、mora/phone、模型选择、配置、恢复、诊断限制和TTS字段。

**R22. 真实门禁。** 无GPU测试、frontend/API、ASR、MFA、mixed、Japanese phenomena、optional Julius和独立 verifier 有明确通过线；全拒收不能让 mixed gate 通过。

## 有序实施计划

### 0. 冻结数据、来源和 gold

所有者：pipeline owner + data owner。依赖：无。

冻结 source manifest、awesome／三个选中项目 commits、当前仓库 revision和 dirty status。准备开发集和调参不可见 held-out：纯日语、纯英语、mixed、读音歧义和日语现象。人工 gold 标 `orig_text`、spoken reading、route、关键 word span、language seam sample；不要求逐 phone 手标。

### 1. Supply-chain lock 与隔离环境

所有者：environment／license owner。依赖：0。

实现 lock schema和 verifier；从 pinned pyopenjtalk-plus commit 构建 wheel，记录 wheel SHA-256和 distribution ownership。验证环境中没有第二个 `pyopenjtalk` provider。代码许可和每个字典／模型／binary许可分别审核。Julius资产未齐时保持 diagnostic disabled。

### 2. Schema、状态机和独立 verifier 骨架

所有者：schema／audit owner。依赖：0。

先实现所有 schema、stable error taxonomy、atomic writer和exact-set partition。verifier使用独立解析路径，覆盖hash篡改、extra file、symlink、重复alias、倒置时间、未知schema和错误COMPLETE。

### 3. 音频 inventory 与保守 transform

所有者：audio owner。依赖：2。

实现WAV header、源hash、train/alignment transform和整数sample映射。以真实促音、清化元音fixture证明低能量不会被删除；处理前后source不变，输出transform可重算。

### 4. ASR provider与family-aware reading selector

所有者：ASR／linguistic QA。依赖：2、3。

实现三种profiles、provider receipts、family去重、共同normalizer、reading候选和固定选择层级。单模型／单家族路径只能dev；同一family多个模型不会增加票数。ASR失败、同字G2P和medoid均有明确状态。

### 5. pyopenjtalk-plus contract

所有者：frontend owner。依赖：1、2、4。

实现provider capability probe、所有显式flags、caller span调用和raw输出。针对`g2p_mapping`、三种`make_phoneme_mapping`输入模式、`(0,0)`、leading long mark、标点／空白、tsqyomi和Marine写golden tests。任何坐标域不明都拒绝。

### 6. Semantic graph、MFA adapter与aliases

所有者：frontend／phonology owner。依赖：5和模型inventory。

实现mora tokenizer和semantic nodes，再实现Japanese MFA v3 adapter。以配套词典抽样／golden rows验证roundtrip，覆盖sakura、Tōkyō、促音、拨音、清化、拗音、yotsugana。生成per-occurrence alias、locked dict和反向map；所有phone预检inventory。

### 7. 双 MFA runners与mixed merge

所有者：MFA／alignment owner。依赖：3、4、6。

完成Japanese runner，复用English runner的语言无关隔离能力但不改变旧CLI。实现dual Qwen pass、unit span聚合、language runs、sample offset、seam retry/reject和strict ledgers。禁止phone clipping和旧中文phonemap。

### 8. Julius diagnostic backend

所有者：diagnostic owner。依赖：1、3、6。

在独立环境实现converter、16k PCM16、`-palign`、lab parser和semantic差异报告。先测试它不能写production namespace，也不能改变reading selector/MFA hash；实际Julius smoke只有资产许可和hash完整后运行。

### 9. TTS exporter、韵律接口与cache

所有者：TTS data／pipeline owner。依赖：7，可选8。

导出alignment-v2、TextGrid和training-record-v1，保留native/semantic/mora graph及masks。实现完整stage identity、checkpoint/resume、fresh publish和独立verify gate；Julius开关只改变diagnostic identity，不改变MFA内容hash。

### 10. 文档、回归、canary与发布

所有者：QA／release owner。依赖：1-9。

补用户文档和示例配置。先无GPU测试、现有回归，再frontend真实smoke、ASR、MFA、mixed和可选Julius。所有阈值只在开发集确定；held-out一次性生成结果报告。未过的能力保持disabled／NO-GO，不用人工删除失败样本美化比例。

## Canary矩阵和初始工程门槛

| 集合 | 样例／数量 | 人工gold | 通过条件 |
|---|---|---|---|
| frontend基础 | `さくら`、`東京`、普通假名、标点 | caller span、reading、mora | span exact；`さくら` 3 mora；`東京` 4 mora；无`(0,0)` lexical span |
| 长音表示 | `とーきょー` | 4 mora和三表示关系 | Julius/OpenJT/MFA通过semantic edges关联；不直接zip；不产生伪内部时间 |
| 促音／拨音／清化 | `学校`类促音、`ン`、真实清化样本 | reading、保留事件 | 0个被当silence删除；inventory/roundtrip通过 |
| 读音歧义 | 同汉字不同读音、同surface双occurrence | 每occurrence reading | alias不同且各自dict恰一行；MFA不替selector投票 |
| frontend API | caller/morph-only/NJD、leading长音符 | 坐标域／期望拒绝 | caller模式通过；其它域不冒充origin；unbound稳定拒绝 |
| ASR baseline | 至少30条脚本一致、30条读音歧义 | family、surface、kana | 一家族一票；qwen-only不发布；同字shared-G2P标记100% |
| pure JA/EN | 各10条 | words、route、关键边界 | JA只来自Japanese MFA，EN只来自ARPA；phone越inventory=0 |
| mixed held-out | 40条：日→英、英→日、无停顿、短英文贴日语助词各10条 | route、word spans、seam sample | 自动接受≥34/40；每桶≥8/10；seam MAE≤80ms、P95≤160ms；route error、phone clipping、cross-language overlap、错误COMPLETE均0 |
| Latin语义 | `AI/USB`日语读、真英语、日语化借词 | route、reading | policy/override一致；未知保持unresolved |
| Julius隔离 | 至少5条JA，含长音和促音 | diagnostic availability | 有资产时产出diagnostic；MFA/TTS hashes开关前后相同；无资产时不fallback |
| TTS schema | 所有accepted canary | audio/phone/mora/mask | durations与sample一致；mora graph完整；accent/F0来源和mask合法 |

这些数值是第一版工程验收线，不是已测性能声明。人工 seam／word gold 允许存在；禁止的是伪造逐 phone 真值或把估算 mora 内部边界写成真实 target。

## 要求到验收追踪

| 要求 | 可观察验收标准 | 验证步骤／命令 |
|---|---|---|
| R1 | runtime不import/clone awesome；lock只记录固定commit | `test_ja_supply_chain.py`；检查dependency graph |
| R2 | 每项有revision/hash/license scope；模型许可不继承代码许可；未知许可阻断 | `verify_ja_supply_chain.py --strict`；license mutation tests |
| R3 | 源hash不变；train/alignment sample可逆；促音／清化fixture不删除 | `test_ja_audio_axis.py`；independent verifier |
| R4 | 三profile行为固定；同family双模型仍一票 | `test_ja_asr_profiles.py` provider/family matrix |
| R5 | 选择层级可重放；shared-G2P有标志；MFA/Julius vote count恒为0 | `test_ja_reading_selector.py`；evidence audit |
| R6 | 仅一个distribution提供namespace，commit/wheel/API吻合 | frontend capability command；双provider负例 |
| R7 | caller spans exact；lexical `(0,0)`／越界／倒置拒绝 | `test_ja_frontend_spans.py` 三输入模式矩阵 |
| R8 | 五个选项显式且进入identity；切换任一项使cache miss | `test_ja_frontend_options.py`；resume mutation |
| R9 | Tōkyō三表示通过semantic graph关联；无字符串replace/索引zip | `test_ja_phone_adapter.py` golden／negative fixtures |
| R10 | receipt显示Japanese v3.0.0、MFA3.3.9、asset/inventory/license；smoke通过 | model check、minimal validate/align、asset tamper tests |
| R11 | EN仅ARPA，JA仅Japanese；native phone namespace隔离 | dual runner tests；inventory mutation |
| R12 | 相同surface不同reading有不同alias；每alias唯一dict行和反向map | `test_ja_locked_aliases.py`；verifier重算 |
| R13 | pure/mixed输出合法；失败seam只retry/reject，不裁phone | `test_ja_en_seams.py`；40条mixed gate |
| R14 | mora graph多对多；长音无伪内部真值 | `test_ja_mora_graph.py`；Tōkyō fixture |
| R15 | Julius开关不改变MFA/TTS hashes；不进入reading votes | `test_julius_diagnostic_isolation.py`；namespace audit |
| R16 | accent/F0字段来源分离，unknown/devoiced有mask | `test_ja_prosody_schema.py` |
| R17 | TTS phones/durations与global samples一致，text/reading/speaker完整 | `test_ja_tts_export.py`；independent verifier |
| R18 | 任一source/model/flag/mapping/config/schema变化拒绝旧cache | `test_ja_resume_identity.py` mutation matrix |
| R19 | expected精确分区；PARTIAL无COMPLETE；每失败有code/path | `test_ja_accounting.py`；manifest audit |
| R20 | 现有MFA/Qwen/English回归全过；默认下载行为相同 | 现有回归命令；download snapshot test |
| R21 | 中文文档含sakura/Tōkyō、模型、CLI、恢复、诊断边界 | docs checklist；示例config schema check |
| R22 | canary各行过门槛且mixed不是全拒收；独立verifier返回0 | rollout report verifier；publish receipt |

## 计划验证命令与预期

以下命令都属于后续实现窗口；当前仓库尚无这些新增脚本。

### Supply chain 与 frontend

```bash
cd /mnt/local_E/MFA_Pause/repo
python scripts/verify_ja_supply_chain.py \
  --lock third_party/japanese_nlp_sources.lock.json --strict
python scripts/check_ja_frontend.py \
  --config configs/japanese_english_tts.yaml --no-write
python -m pytest -q \
  tests/test_ja_supply_chain.py \
  tests/test_ja_frontend_spans.py \
  tests/test_ja_frontend_options.py \
  tests/test_ja_phone_adapter.py \
  tests/test_ja_mora_graph.py
```

预期：全部返回0；`--no-write` 不创建workspace；namespace provider唯一；commit/wheel/API/flags一致；所有golden spans、mora graph和inventory通过。

### ASR、MFA、mixed、diagnostic 与TTS

```bash
cd /mnt/local_E/MFA_Pause/repo
python scripts/run_ja_en_pipeline.py \
  --config configs/japanese_english_tts.yaml --check
python -m pytest -q \
  tests/test_ja_asr_profiles.py \
  tests/test_ja_reading_selector.py \
  tests/test_ja_locked_aliases.py \
  tests/test_align_japanese_mfa.py \
  tests/test_ja_en_seams.py \
  tests/test_julius_diagnostic_isolation.py \
  tests/test_ja_tts_export.py \
  tests/test_ja_resume_identity.py \
  tests/test_ja_accounting.py
python scripts/run_ja_en_pipeline.py \
  --config /data/canary/ja_en_tts.yaml --stage inventory,asr,reading,frontend
python scripts/run_ja_en_pipeline.py \
  --config /data/canary/ja_en_tts.yaml --stage align,merge,tts,verify
python scripts/verify_ja_en_tts.py --workspace /data/runs/ja_en_canary
```

预期：所有unit有可解释状态；phone inventory越界为0；aliases一对一；mixed达到R22阈值；Julius disabled时无production变化；verifier返回0后才写COMPLETE。

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
python -m compileall -q scripts tests
git diff --check
```

预期：全部通过；中文默认steps、Chinese-only anchored profile、strict-en v2和默认model download清单语义不变。

## 可观察验收标准

- 用户可从一个manifest运行 raw game audio → ASR blind evidence → reading selection → JA/EN MFA → TTS JSONL，并看到逐stage receipts。
- awesome repository 不出现在runtime imports；选中资源全部有固定revision和独立许可／hash记录。
- frontend的每个caller span可回切原文本；normalized/NJD spans不会冒充origin；非空lexical `(0,0)`不会发布。
- `さくら`报告3 mora；`東京`报告4 mora，并能展示Julius/OpenJT/MFA三表示关系而不声称一一phone等价。
- 每个Japanese/English occurrence都用唯一alias；同surface不同reading不会互相覆盖。
- 纯JA phones只来自Japanese MFA，纯EN phones只来自ARPA；mixed seam没有clip、跨语言overlap或越WAV。
- ASR evidence按family计票；同family重复、Qwen anchor、MFA或Julius均不增加reading票数。
- Julius开关前后production MFA/TTS内容hash一致；Julius只有diagnostic namespace。
- accent prediction与measured F0字段不同，未知／清化／无声mask可审计。
- mixed held-out达到34/40和每桶8/10，seam MAE／P95达标；全拒收无法通过。
- 完全相同identity可在加载模型前resume；任一source、model、wheel、flag、adapter或config变化使cache失效。
- 现有中文和English回归通过，且元数据列出的其他工作区改动未被覆盖。

## 风险、缓解与回滚

| 风险 | 后果 | 缓解 | 回滚 |
|---|---|---|---|
| 把awesome目录当依赖或最新版 | 不可复现、许可混乱 | 固定index commit，只生成lock | 删除lock中的选型项，不影响runtime |
| pyopenjtalk namespace冲突 | import到错误实现 | 独立env、single provider、wheel/API receipt | 关闭frontend production，保留raw/ASR artifacts |
| char span坐标域混淆 | token绑定错原文 | 强制caller_text模式，三坐标域字段，`(0,0)`拒绝 | 回退到上一个frontend lock并全量cache miss |
| tsqyomi／long-vowel flags漂移 | reading或phone静默改变 | flags显式、进入identity、golden diff | 恢复固定flags并重跑frontend后续stage |
| semantic mapping凭字符串猜 | 错把MFA `c`等同其它体系 | graph+model adapter+dictionary roundtrip+inventory | 阻塞相关occurrence，不添加未知phone |
| ASR同家族重复被当独立 | reading置信度虚高 | family vote cap、shared-G2P标志 | 重新生成reading evidence，后续cache失效 |
| Japanese MFA/runtime不兼容 | align命令或输出失败 | isolated capability smoke和真实assets receipt | 保持现有MFA环境不动，另建实验env |
| mixed anchor吞邻语言 | 错误phone边界 | dual pass、ownership、双侧retry、held-out gate | `mixed.enabled=false`，纯JA/EN仍可用 |
| Julius结果回流production | 双重时间权威 | namespace/receipt/hash isolation tests | 删除diagnostic目录，production hash保持 |
| 促音／清化被能量删除 | 日语时长target错误 | 禁止词内低能量自动删除，真实fixtures | 恢复raw alignment和transform |
| 文本accent被当F0 | 训练标签含伪声学真值 | provenance和独立mask | 清除预测字段，不动phone timing |

所有新产物在fresh workspace；旧入口默认不变。回滚只移除新入口、独立环境、opt-in下载组和新schema读写，不删除历史run，不修改用户现有dirty文件。

## 未决门禁

| 门禁 | 证据／缺口 | 影响 | 责任人 | 解除路径 |
|---|---|---|---|---|
| G1 frontend build/API | commit和API文档已固定，目标平台wheel尚未验证 | 阻塞production frontend | environment owner | reproducible wheel hash、single provider、全部API/span canary通过 |
| G2组件许可 | package源码MIT已核，底层字典／可选模型许可分开 | 阻塞相应artifact发布 | license owner | lock中每项license hash/status=approved |
| G3 ASR revisions/预算 | 模型名和family已定，精确revision/许可/GPU预算未冻结 | 阻塞baseline reading gate | ASR owner | 三家族model receipts和30+30 canary通过 |
| G4 Japanese MFA资产/runtime | 固定v3.0.0 acoustic URL/92,191,596 bytes及dictionary URL/21,264,022 bytes；byte size不是完整性证明且无官方digest；当前runtime3.3.9 | 阻塞真实JA alignment | MFA owner | 从上文官方URL下载、计算并冻结SHA-256、license核验、minimal validate/align、inventory receipt |
| G5 semantic adapter覆盖 | 东京条目证明差异，不证明全inventory | 阻塞未覆盖phones | frontend/phonology QA | 全inventory分类、golden dictionary roundtrip、未知集合为空 |
| G6 mixed code-switch | 官方未保证同句JA/EN anchor准确率 | 阻塞mixed publish | ASR/alignment QA | 40条held-out达到34/40、每桶8/10、误差和零严重错误门槛 |
| G7 Latin policy lexicon | `AI/USB`可日语读或真英语 | 阻塞歧义occurrence | data owner | 提供带source_id的route/reading overrides或接受unresolved |
| G8 Julius资产 | 代码commit已定，binary/model/dict和许可未定 | 只阻塞optional diagnostic | diagnostic owner | 独立assets receipt和5条smoke；不影响MFA GO |

G1-G8不阻塞schema、lock、fixtures、error taxonomy和independent verifier开发。G8永不阻塞production MFA；G6只阻塞mixed；G7只阻塞对应occurrences。

## 执行检查清单

- [ ] 进入 `/mnt/local_E/MFA_Pause/repo`，重读本交接和`CLAUDE.md`，记录HEAD/status并保护已有改动。
- [ ] 创建fresh feature worktree和fresh run workspace；不复用旧alignment/cache。
- [ ] 先实现supply-chain lock、schema、errors、集合守恒和独立verifier骨架。
- [ ] 构建并固定pyopenjtalk-plus wheel，证明single namespace provider和API capabilities。
- [ ] 把所有frontend options显式化，完成caller span／`(0,0)`／ignored语义tests。
- [ ] 完成三ASR家族baseline和family-aware reading evidence；qwen-only保持dev。
- [ ] 获取Japanese/English v3.0.0 assets receipt，完成MFA3.3.9真实smoke。
- [ ] 完成semantic graph、全inventory adapter、per-occurrence aliases和locked dictionaries。
- [ ] 完成JA/EN runners、dual anchors、global sample轴和seam retry/reject。
- [ ] 在production已独立通过后再实现optional Julius diagnostic隔离。
- [ ] 导出TTS JSONL/TextGrid，验证mora graph、durations、accent/F0 masks。
- [ ] 运行无GPUsuite和全部现有回归，保存versions/commands/results。
- [ ] 开发集冻结阈值后一次性跑held-out；未过门槛不打开mixed。
- [ ] independent verifier返回0后才写COMPLETE和fresh publish。
- [ ] 若发现并修复旧逻辑冲突，按`CLAUDE.md:1-24`补Regression Archive；纯新增不追加case。

## 就绪判断

**实现规划：GO。真实production：按能力分级NO-GO。**

可立即开始：supply-chain lock、schema、ASR family evidence、frontend contract、semantic graph、aliases、verifier和无GPUfixtures。Japanese production需G1-G5通过；mixed需额外G6-G7；optional Julius只受G8约束。任何门禁未过都已有可观察状态和决策路径，不需要实施者自行猜兼容性或默许fallback。

模型决策保持明确：生产日语是 Japanese MFA acoustic v3.0.0 + dictionary v3.0.0；真英语是 English US ARPA v3.0.0；pyopenjtalk-plus是候选文本前端；Julius4seg/JATTS是可选诊断；MFA始终是production phone timing authority。

## 下一执行窗口启动说明

1. `cd /mnt/local_E/MFA_Pause/repo`。
2. 阅读 `handoffs/20260921T050932Z-awesome-japanese-nlp-ja-en-mfa-pipeline.md`；无需依赖前版才能执行。
3. 执行 `git rev-parse HEAD && git branch --show-current && git status --short`。若HEAD或引用文件已变化，刷新行号、hash和受影响决策；保留所有列出的已有改动。
4. 从实施计划0-2开始；第一份review应只包含lock/schema/errors/verifier/fixtures，不先下载大模型或修改中文step order。
5. 第二阶段完成隔离frontend wheel、ASR evidence和semantic adapter；第三阶段才接MFA/mixed；Julius最后加入且保持optional。
6. 默认配置必须保持`mixed.enabled=false`、`julius_diagnostic.enabled=false`。只有对应held-out／assets门禁通过并留下receipt后才能分别开启。
7. 每个提交运行追踪矩阵中对应命令；最终运行现有回归、independent verifier和fresh publish检查。
