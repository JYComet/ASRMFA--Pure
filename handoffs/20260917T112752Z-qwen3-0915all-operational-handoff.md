# Qwen3 0915ALL 全语料项目运行与续作交接

## 元数据

- UTC 快照时间：`2026-09-17T11:27:52Z`
- 任务标识：`qwen3-0915all-operational-handoff`
- 仓库：`/mnt/local_E/MFA_Pause/repo`
- 分支：`codex/0915all-full-corpus`
- 修订：`e09826b1d38644bedfb5ed69f93ce083f9ba6c6c`
- 工作树：脏；`git status --porcelain=v1` 共 67 行，快照 SHA-256 为 `df12465fa024a5bb5869adf07dcd5e9cff7607f3e9017c0e19b9a6ae1bd73ad6`
- 生产配置：`/mnt/local_E/MFA_Pause/repo/configs/qwen3_0915all_full_20260914.yaml`
- 生产运行根目录：`/mnt/nvme3/qwen3_0915all_full_20260914`
- TextGrid 输出：`/mnt/Raw/0915ALL`
- 游戏补静音音频：`/mnt/Raw/GAMESL`
- 人工抽样：`/mnt/Raw/0917`
- 本交接文件：`/mnt/local_E/MFA_Pause/repo/handoffs/20260917T112752Z-qwen3-0915all-operational-handoff.md`
- 规划路由：由根代理显式调用 `gpt-5.6-sol`、`reasoning_effort=high` 生成规划草稿，根代理复核证据并写入本文件。

工作树中的 `.gitignore`、环境文件、`scripts/run_pipeline.py`、若干模型文件删除记录等状态在本交接创建前已经存在。接手者必须保留这些状态，不能执行 `git reset --hard`、`git clean`、整树 checkout 或批量恢复。

## 目标与任务内容

项目目标是将主管线改为 Qwen3-only：不再使用 NVASR/FunASR 生成文本或时间戳；无参考文本时由 Qwen3 ASR 生成文本并由 Qwen3 Forced Aligner 生成时间戳，有参考文本时直接以归一化参考文本驱动 Qwen3 Forced Aligner。Qwen3 时间戳先经过固定正则化，再作为 MFA 与后处理的基础时间轴；后续边界修正可以修改最终时间轴，发布结果是修正后的 TextGrid。

本轮全量处理覆盖 GAMEDATA、指定的 v5_0707 说话人和鸣潮中文数据。GAMEDATA 与鸣潮先将首尾静音补到 0.5 秒并持久化到 GAMESL，再进入后续管线；v5_0707 不补静音，也没有在后续名称核对中被改动。最终输出按说话人发布，游戏说话人增加游戏前两字拼音首字母前缀，避免跨游戏同名角色合并。

## 当前状态与权威统计

生产已结束，不是仍在运行。`/mnt/nvme3/qwen3_0915all_full_20260914/status.json` 的状态为 `complete_with_failures`，77 个计划块全部为 `complete`。该状态表示全批完成且保留逐条过滤/失败记录，不表示编排器崩溃。

`final_report.json` 的权威计数：

| 类别 | 数量 | 解释 |
|---|---:|---|
| input | 760,927 | 冻结后进入处理口径的输入 |
| accepted | 721,483 | 正式输出 |
| filtered | 28,415 | 保存于 `_filtered` 供复查 |
| producer_failure | 593 | Qwen producer 过滤 |
| pipeline_or_mfa_failure | 10,436 | 分阶段准备或管线失败 |
| excluded | 21,536 | 依据数据选择规则预先排除，不计入 input |
| invalid | 0 | 冻结清单中的 invalid 计数 |

正式结果与复查结果合计保留 749,898 条；未形成 TextGrid 的 input 条目为 11,029 条。`pipeline_or_mfa_failure` 的组成是：9,139 条音频采样率异常、含非有限采样或无语音；1,246 条参考文本归一化后没有可发音词法单元；51 条含当前规则不支持的希腊、俄文、注音或日文字符。另有 593 条原因是 `qwen_producer_filtered`。逐字符明细以 `final_report.json.reasons` 为准。

## 当前数据流

1. `full_corpus_inventory.py` 扫描三类数据，记录源路径、源哈希、参考文本、模式、时长和稳定 `run_stem`，再冻结为 `frozen_inventory.json`。
2. `full_corpus_orchestrator.py prepare` 生成 77 个块与分模式配置；reference 块有参考文本，fallback 块无参考文本。
3. `full_corpus_stage.py` 在每条进入管线前复核冻结哈希。GAMEDATA/Wuwa 归一化为单声道 PCM16、首尾至少 0.5 秒静音，并发布到 `GAMESL/游戏/原说话人/run_stem.wav`；v5 保持原有音频轴和样本表示。
4. `qwen3_prealign.py` 仅加载 Qwen3。reference 模式不调用 ASR 文本生成，直接用参考文本；fallback 模式调用 Qwen3 ASR。两种模式都用 Qwen3 Forced Aligner 产生词级时间戳。
5. `qwen3_timestamp_normalization.py` 固定归一化文本、空隙和停顿证据，保留原始模型坐标。
6. `run_pipeline.py` 继续执行已有的 normalize、adjust、中文 MFA、英文 MFA 与后处理逻辑。MFA 的全套逻辑保留，输入锚点改为 Qwen3 时间戳。
7. `postprocess_textgrids.py` 对最终边界、能量延伸、短词、异常静音、标点投影和过滤执行既有处理。
8. `full_corpus_publish.py` 验证五层 TextGrid、音频轴、Qwen 身份、时间戳正则化和标点证据，再原子发布到 `0915ALL`，同时保存逐块回执和回滚证据。

`mfa.native_anchor_fallback: true` 是 stock MFA 兼容路径：它把已经生成的 Qwen 词锚点构造成 MFA 可接受的分段语料，相关实现见 `run_pipeline.py:189-241`。它不重新启用 NVASR/FunASR，也不把后处理边界回退到旧 ASR 时间轴。不要仅因字段名含 `fallback` 就删除这条兼容逻辑。

## 固定时间戳和文本规则

- 输入英文逗号、句号、感叹号和问号映射为中文规范标点；`~`、全角波浪线和转义形式 `\~` 归一为 `…`。
- 装饰性引号、括号等异常标点删除；参考文本最终仅保留中文、英文字母、空格及 `，。！？…、`。
- 连续多个 `…` 归一为一个 `…`。
- `<sp0>` 至 `<sp3>` 是时间注记，不作为 MFA 发音词，也不能覆盖原有标点。
- 开头空隙始终保留为边界静音，不生成省略号。
- 非开头、短于 200 ms 的空隙等价于 sp0：前一个词向后延伸覆盖该空隙。
- 非开头、达到 200 ms 的空隙保留该边界已有标点；只有边界无标点时才补 `…`。
- 停顿等级为：小于 500 ms 是 sp1；500 ms 至小于 1,500 ms 是 sp2；达到 1,500 ms 是 sp3。
- `original_start_s`、`original_end_s` 和 raw 坐标是模型原始证据；归一化/后处理坐标是可修改并最终发布的时间轴。禁止删除原始证据后声称可回溯。
- 最终 accepted TextGrid 层级契约是 `raw_text`、`pinyin`、`hanzi`、`words`、`pinyin_phones`。

## 输入选择与输出结构

配置证据见 `configs/qwen3_0915all_full_20260914.yaml:1-102`：

- GAMEDATA：`/mnt/Raw/GAMEDATA`。
- v5_0707：`/mnt/Raw/v5_0707`；保留 `Xuehusang`、`Xiaoyuan`、`Wumi`、`Mieli`、`合成ria` 以及 `GS`、`HK`、`SR`、`WW` 前缀目录。
- Wuwa：`/mnt/Raw/Onlinedataset/鸣潮/中文/WutheringWaves2.2_CN/中文 - Chinese`；排除“其它语音 - Others”和“带变量语音 - Placeholder”。
- 正式 TextGrid：`/mnt/Raw/0915ALL/公共说话人/run_stem.TextGrid`。
- 过滤 TextGrid：`/mnt/Raw/0915ALL/_filtered/公共说话人/run_stem.TextGrid`。
- 游戏音频：`/mnt/Raw/GAMESL/游戏/原说话人/run_stem.wav`。

GAMESL 已有游戏层级，因此保留原说话人目录；扁平的 0915ALL 必须用公共说话人命名空间。规则为游戏名最前两个汉字的拼音首字母大写，例如崩坏三姬子为 `BH姬子`，崩铁姬子为 `BT姬子`，鸣潮今汐为 `MC今汐`。若说话人已经以前缀开头，不重复添加。v5 没有游戏字段，名称保持不变。

实际扫描得到 1,144 个游戏与说话人组合，应用规则后仍有 1,144 个唯一公共名称，冲突为 0。原始跨游戏同名共有 4 组：可可利亚、姬子、希儿、瓦尔特，均同时出现在崩坏三和崩铁。

## 运行文件与追溯入口

运行根目录 `/mnt/nvme3/qwen3_0915all_full_20260914` 中的证据用途：

| 路径 | 用途 |
|---|---|
| `frozen_inventory.json` | 760,927 条输入、源路径、哈希、reference/fallback 模式、稳定 stem；约 553 MB，避免无目的整文件反复加载 |
| `chunks.json` | 77 个块的冻结计划与条目归属 |
| `status.json` | 每条终态、失败原因和每块当前终态 |
| `events.jsonl` | 启动、预取、完成、发布、重试的追加事件流 |
| `final_report.json` | 最终计数、原因、分组、终态 stem 和发布回执；约 415 MB |
| `preflight_receipt.json` | 依赖、磁盘、GPU、配置和 Wuwa 分母门禁 |
| `canary_receipt.json` | reference/fallback 小样本验收证据 |
| `prepare_status.json` | 全量 Qwen/时间戳准备阶段状态 |
| `speaker_namespace_migration.json` | 已发布结果的说话人前缀迁移统计 |
| `configs/` | 每块解析后的生产配置 |
| `logs/` | 每块 `prealign`、`prepare`、`downstream` 日志 |
| `chunks/CHUNK_ID/gamesl_publication.json` | GAMESL 音频发布目标、哈希、替换和回滚绑定 |
| `chunks/CHUNK_ID/output_publication.json` | accepted/filtered TextGrid 发布目标、哈希和说话人绑定 |
| `rollback/` | 原子替换前的旧文件；只能按具体发布回执使用 |
| `monitor_probe.py` | 运行期间的只读状态探针 |
| `status.pre-config-migration-1789422723454443777.json` | 配置迁移前状态快照 |

样本目录 `/mnt/Raw/0917` 含 10 组同基名 WAV/TextGrid；`manifest.json` 记录源路径、复制路径、大小和 SHA-256。已验证 10 个 WAV、10 个 TextGrid、基名集合相等，且源与副本哈希相等。

## 历史问题和恢复情况

### 单条异常导致整块失败放大

第 10 个 reference 块 `reference-181b97dad0777e0f` 曾在下游 normalize 阶段因 `u5250d20755f931d47755c4e284c4443d` 的畸形 words 区间失败。当时一个条目导致整块 9,768 条暂时标记为 `pipeline_failure`。编排器随后按证据恢复并补跑；当前 `status.json.terminal` 中该块为 `complete`，且存在 `gamesl_publication.json` 与 `output_publication.json`。`events.jsonl` 有 92 条 `chunk_terminal` 事件但只有 77 个唯一块，这是恢复和重试记录，不是 15 个重复生产块。

### NAS 等待被误判为假死

staging/prefetch 阶段主要进行 NAS 到 NVMe 的复制，可能没有 GPU 子进程。事件中的 `prefetch_started`、`prefetch_completed`、目录最近写入时间以及 `monitor_probe.py` 才是这段时间的判定依据。后续实现允许处理当前块时预取下一块，减少纯等待。

### 说话人迁移

在最初 27 个已完成块上执行了前缀迁移：184,990 条移动、78,857 条保持不变、0 条去重冲突。迁移后旧的无前缀同名目录已清理；后续块由发布代码自动使用前缀。完整证据在 `speaker_namespace_migration.json` 和每块 `output_publication.json`。

### GAMESL 与 TextGrid 配对

逐回执与磁盘实存审计结果：393,757 个游戏 TextGrid 均有同 basename GAMESL 音频，缺失 0、名称不一致 0、重复 0；实际检查 787,514 个文件路径，缺失 0。其中 accepted 游戏 TextGrid 369,311 条，filtered 游戏 TextGrid 24,446 条。GAMESL 总计发布 394,197 条音频，因此另有 440 条 WAV 没有 TextGrid；这 440 条全部是 `qwen_producer_filtered`。从 TextGrid 查音频是 100% 完整的，但整个 GAMESL 目录不是严格双向一一对应。

### v5 名称核对

曾讨论让 v5 TextGrid 改回原音频名，但用户明确限定本次一一对应问题只指 GAMESL。随后只读演练被停止，未使用 apply，相关临时代码修改已撤销，测试恢复通过，v5_0707 内容没有被修改。

## 代码与符号证据

| 文件与行 | 责任 |
|---|---|
| `scripts/full_corpus_inventory.py:136-152` `_make_item` | 生成稳定 stem、记录参考模式与是否需要补静音 |
| `scripts/full_corpus_inventory.py:208-300` `scan_sources`、`write_frozen_inventory` | 选择、并行扫描、冲突检查、不可变库存 |
| `scripts/full_corpus_stage.py:139-281` `stage_item` | 源哈希复核、参考文本归一、GAMESL 补静音及回执 |
| `scripts/qwen3_prealign.py:376-446`、`484-555` | reference/fallback 路由、八卡 Qwen 设置与身份 |
| `scripts/qwen3_prealign.py:646-781` | 时间戳归一、输出 bundle 和 Qwen-only receipt |
| `scripts/qwen3_timestamp_normalization.py:16-38` | schema、规范标点与支持字符常量 |
| `scripts/qwen3_timestamp_normalization.py:41-80` `normalize_qwen_input_text` | 表面标点归一 |
| `scripts/qwen3_timestamp_normalization.py:132-196` `normalize_qwen_reference_text` | 参考文本、数字、异常字符归一 |
| `scripts/qwen3_timestamp_normalization.py:236-341` `normalize_timestamps` | 句首静音、前词延伸、标点优先、sp1/2/3、原坐标证据 |
| `scripts/run_pipeline.py:189-241` `_build_native_anchor_corpus` | 将 Qwen 锚点转换为 stock MFA 可接受语料 |
| `scripts/full_corpus_orchestrator.py:1103` `run_prepare_all` | 全量先准备 Qwen 正则化时间戳 |
| `scripts/full_corpus_orchestrator.py:1275` `run_full` | 可恢复串行执行、预取和发布 |
| `scripts/full_corpus_orchestrator.py:1773-1840` `audit` | 终态守恒、回执与目标哈希审计 |
| `scripts/full_corpus_orchestrator.py:1843-1868` `main` | prepare/preflight/canary/prepare-all/run/audit 命令入口 |
| `scripts/full_corpus_publish.py:115-174` `validate_chunk_result` | accepted/filtered/failed 分桶与证据校验 |
| `scripts/full_corpus_publish.py:177-283` `_validate_evidence` | Qwen-only、时间戳与标点证据门禁 |
| `scripts/full_corpus_publish.py:313-369` `publish_chunk` | 原子发布与 rollback |
| `scripts/full_corpus_publish.py:372-420` `build_final_report` | 守恒计数和最终报告 |
| `scripts/speaker_namespace.py:13-32` | 两汉字拼音首字母前缀及幂等规则 |
| `scripts/migrate_publication_speaker_prefixes.py:56-116` | 历史发布路径与回执迁移 |

相关测试集中在 `tests/test_qwen3_timestamp_normalization.py`、`tests/test_qwen3_prealign.py`、`tests/test_qwen3_energy_pause_extension.py`、`tests/test_full_corpus_inventory.py`、`tests/test_full_corpus_stage.py`、`tests/test_full_corpus_orchestrator.py`、`tests/test_full_corpus_publish.py`、`tests/test_speaker_namespace.py` 和 postprocess 系列测试。

## 事实、假设、决策与开放问题

### 事实

- Qwen3-only 身份、时间戳正则化 schema、原始时间戳和标点投影是发布门禁。
- 77/77 块完成，最终计数守恒；历史整块失败已经恢复。
- 游戏公共说话人规范化后冲突为 0。
- 每个已发布游戏 TextGrid 都能找到同基名 GAMESL 音频。
- 440 个额外 GAMESL WAV 有明确的 producer 过滤原因。
- 本交接新鲜度检查在上述 UTC 快照时通过。

### 假设

- 本交接写入后，没有外部进程改写 NAS/NVMe 运行目录。
- `/mnt/Raw` 仍映射到用户所述 NAS 共享位置。
- 现有 67 行工作树状态均需先归属审查，不能默认属于本任务。

### 已定决策

- 保留 MFA 全套逻辑和时间戳锚点修正；不恢复 NVASR/FunASR。
- 原始时间轴作为证据，最终时间轴允许后续逻辑修改，发布最终修改结果。
- 保留已有标点；只在长停顿且边界无标点时补省略号。
- GAMESL 保持游戏目录分层；0915ALL 用游戏前缀消除同名说话人碰撞。
- 逐条失败保留原因，不伪造全成功状态。
- 已完成运行默认只读；发现局部问题时按块和发布回执恢复，不全量重跑。

### 开放问题

| 问题 | 证据 | 影响 | 负责人 | 决策路径 |
|---|---|---|---|---|
| 是否隔离 440 个无 TextGrid WAV | 发布回执与 status 均指向 `qwen_producer_filtered` | 改变 GAMESL 双向配对口径和现有审计基线 | 数据所有者 | 明确保留、移动到隔离目录或删除；操作前生成路径和哈希清单 |
| 是否专项回收 11,029 个未产出条目 | `final_report.json.counts/reasons` | 可能增加可用量，也可能把无声和无词法数据错误放回训练集 | 数据所有者与管线维护者 | 先按 9,139、1,246、51、593 四类分别抽样，再决定独立补跑 |
| 是否提交当前代码状态 | 工作树 67 行且混有模型文件删除 | 错误提交可能吞入用户环境与模型缓存变化 | 仓库维护者 | 逐文件归属审查，只暂存本任务确认文件 |

当前没有阻止只读验收或新会话接手的阻塞。上述三项需要所有者决策后才能执行写操作。

## 范围与约束

### 范围内

- 读取与审计现有代码、配置、运行状态、日志、回执和输出。
- 按失败原因抽样并提出局部补救方案。
- 根据单块证据恢复明确缺失或损坏的发布项。
- 维护 Qwen3-only、固定时间戳正则化和说话人命名规则。

### 范围外

- 未经新决策删除 440 个额外 WAV。
- 未经分类验收重新纳入 11,029 个失败条目。
- 修改、重命名或补静音 v5_0707。
- 清理脏工作树、模型缓存删除或其他用户改动。
- 在没有证据不一致的情况下重跑 760,927 条全量数据。

### 不变量

- 同一个 `run_stem` 在冻结清单、块配置、GAMESL、TextGrid、状态和回执中必须指向同一条数据。
- published target 必须与回执 SHA-256 一致。
- accepted/filtered/失败终态必须完整覆盖 input 且互斥。
- 游戏/Wuwa 时间轴必须与补静音后的 GAMESL 音频一致；v5 时间轴与原音频一致。
- 说话人前缀迁移幂等，不能二次添加。
- 不得以“校验优化”为由跳过发布所需的 Qwen 身份、时间戳、标点和音频轴证据。

## 编号需求

- **R1**：维持 Qwen3-only 的文本与时间戳生产，禁止 NVASR/FunASR 回流。
- **R2**：reference 与 fallback 两种模式按既定职责分流。
- **R3**：每条 Qwen 时间戳必须经过固定正则化并保留原始坐标。
- **R4**：停顿、标点和句首静音必须遵守本交接列出的固定规则。
- **R5**：MFA、英文 MFA、能量边界和后处理继续消费 Qwen 基础时间轴，最终发布后处理时间轴。
- **R6**：GAMEDATA/Wuwa 使用 0.5 秒边缘静音的 GAMESL 音频；v5 不改动。
- **R7**：跨游戏说话人必须使用幂等游戏前缀，规范名称不得碰撞。
- **R8**：全部输入必须进入 accepted、filtered、producer failure 或 pipeline/MFA failure 中唯一一个终态。
- **R9**：发布必须是原子的，并由哈希绑定回执和 rollback 支持局部恢复。
- **R10**：游戏 TextGrid 到 GAMESL 音频必须保持同 basename 且无缺失、无重复。
- **R11**：历史失败、重试、迁移和数据质量问题必须可由持久化证据复核。
- **R12**：任何续作必须保护脏工作树与已完成生产，优先只读验收和局部处理。

## 需求、验收与验证追踪

| 需求 | 可观察验收标准 | 验证步骤 | 预期信号 |
|---|---|---|---|
| R1 | Qwen receipt provider 为 `qwen3_hf`，路由含 forced aligner，不含 nvasr | 检查 `qwen3_prealign.py`、每块 identity/manifest、日志 | provider 与模型树摘要存在，未出现 NVASR 身份 |
| R2 | reference 不依赖 ASR 文本；fallback 有 Qwen transcript | 抽查两类块配置和 prealign manifest | 两类 `reference_mode` 与文本来源一致 |
| R3 | 所有发布条目带 v2 正则化证据并保存 original/raw 坐标 | 运行 timestamp 测试并抽查 token JSONL | schema 为 `qwen3-timestamp-normalization-v2`，原始字段存在 |
| R4 | 句首、200/500/1500 ms 边界及标点优先均正确 | 运行 timestamp 与 energy pause 测试 | 所有边界用例通过 |
| R5 | 最终 accepted TextGrid 五层齐全且轴与对应音频一致 | 运行 publish 测试并抽查 TextGrid | 五层顺序正确，xmax 差不超过代码容差 |
| R6 | 游戏/Wuwa 补 0.5 秒并发布；v5 无 GAMESL 目标 | 检查 stage receipt 与 canary | head/tail gate 通过，v5 receipt 无 gamesl_wav |
| R7 | 1,144 对变为 1,144 唯一名称，前缀不重复 | 运行 speaker namespace 测试并读迁移报告 | 冲突 0，迁移统计匹配 |
| R8 | 760,927 条终态守恒 | 读取 status 与 final report | 四类 input 终态合计 760,927 |
| R9 | 77 份输出回执可解析，目标哈希匹配 | 按块读取 publication receipt 并验证目标 | 无缺失、重复或 digest mismatch |
| R10 | 393,757 对名称与实存一致 | 复用回执构建 stem 映射并只读 stat | 缺失 0、错配 0、重复 0 |
| R11 | 92 个 terminal 事件可归并为 77 个唯一块，chunk 10 最终 complete | 流式统计 events 并读取当前 status | 92、77、complete |
| R12 | 没有清理现有 67 行工作树状态，没有全量重跑 | 操作前后保存 status 快照与进程列表 | 用户改动保留，仅出现获批的局部产物 |

## 续作计划、依赖与责任

1. **建立只读快照**。仓库维护者记录 HEAD、分支、`git status --porcelain=v1` 与状态哈希。依赖：仓库可读。
2. **核对生产终态**。数据发布维护者读取 status、final report、chunks 和 77 份 publication receipt，确认 R8、R9。依赖：运行根目录可读。
3. **复核数据配对**。数据发布维护者按回执复建游戏 stem 映射，确认 R10，并单独列出 440 条 producer 过滤音频。依赖：步骤 2 完成、NAS 可读。
4. **运行代码验收**。管线维护者运行 Qwen、timestamp、stage、orchestrator、publish、speaker 和 postprocess 针对性测试，确认 R1 至 R7。依赖：Python 测试环境可用。
5. **形成新的只读验收记录**。维护者保存命令、UTC 时间、计数和摘要哈希，不覆盖历史 JSON。依赖：步骤 1 至 4 通过。
6. **处理开放问题**。数据所有者分别决定 440 条额外 WAV、11,029 条失败条目和代码提交范围。依赖：只读验收记录已完成。
7. **执行获批的局部写操作**。仅在明确决策后，按单块、单失败类别或单发布项执行；先生成清单和 rollback。依赖：步骤 6 的书面决策。

## 验证命令

### 仓库与状态快照

```bash
cd /mnt/local_E/MFA_Pause/repo
git branch --show-current
git rev-parse HEAD
git status --porcelain=v1 | tee /tmp/qwen3-0915all-git-status.txt
sha256sum /tmp/qwen3-0915all-git-status.txt
```

当前预期为分支 `codex/0915all-full-corpus`、修订 `e09826b1d38644bedfb5ed69f93ce083f9ba6c6c`。工作树非空是已知事实；不要为让命令输出为空而清理。

### 最终计数和块状态

```bash
RUN=/mnt/nvme3/qwen3_0915all_full_20260914
jq '{state,chunk_count,terminal_count:(.terminal|length),terminal_values:(.terminal|to_entries|group_by(.value)|map({state:.[0].value,count:length}))}' "$RUN/status.json"
jq '{counts,reasons}' "$RUN/final_report.json"
```

预期 `state=complete_with_failures`、`chunk_count=77`、`terminal_count=77`、全部 terminal 值为 `complete`，counts 与本交接表一致。

### 历史重试事件

```bash
RUN=/mnt/nvme3/qwen3_0915all_full_20260914
jq -r 'select(.event=="chunk_terminal") | .chunk' "$RUN/events.jsonl" > /tmp/qwen3-terminal-chunks.txt
wc -l /tmp/qwen3-terminal-chunks.txt
sort -u /tmp/qwen3-terminal-chunks.txt | wc -l
jq -r '.terminal["reference-181b97dad0777e0f"]' "$RUN/status.json"
```

预期依次得到 92、77 和 `complete`。

### 针对性测试

```bash
cd /mnt/local_E/MFA_Pause/repo
python -m pytest -q \
  tests/test_qwen3_timestamp_normalization.py \
  tests/test_qwen3_prealign.py \
  tests/test_qwen3_energy_pause_extension.py \
  tests/test_full_corpus_inventory.py \
  tests/test_full_corpus_stage.py \
  tests/test_full_corpus_orchestrator.py \
  tests/test_full_corpus_publish.py \
  tests/test_speaker_namespace.py
```

预期全部通过。若失败，先判断是环境缺少依赖还是代码回归；不得用跳过测试代替结论。

### 0917 抽样

```bash
jq '{schema,count,pairs:[.pairs[]|{stem,game,speaker,copied_audio,copied_textgrid,audio_sha256,textgrid_sha256}]}' /mnt/Raw/0917/manifest.json
find /mnt/Raw/0917 -maxdepth 1 -type f -name '*.wav' | wc -l
find /mnt/Raw/0917 -maxdepth 1 -type f -name '*.TextGrid' | wc -l
```

预期 count 为 10，WAV 与 TextGrid 各 10 个且每对 basename 相同。若需要验证内容，按 manifest 重算源文件和副本 SHA-256。

`python scripts/full_corpus_orchestrator.py audit --config configs/qwen3_0915all_full_20260914.yaml` 会在 reports_root 写入报告副本，不属于纯只读命令。只有在确认允许刷新 `/mnt/Raw/0915ALL/_reports` 时才运行。

## 风险与回滚

- **工作树污染风险**：批量暂存或恢复会混入环境与模型缓存变化。回滚方式是停止操作，使用步骤 1 的状态快照逐文件归属审查；不要使用整树命令。
- **全量重跑风险**：会覆盖已验收输出并制造新的时间戳差异。回滚方式是取消重跑，以具体块回执定位需要恢复的最小集合。
- **NAS 元数据延迟**：逐文件 stat 数十万文件耗时长，GPU/CPU 可能为 0。使用事件、目录写入时间和进程列表判断，不把等待直接判为死锁。
- **发布覆盖风险**：发布器会为既有目标建立 rollback；若手工移动文件则失去绑定。任何移动前先记录路径、旧/新哈希和对应回执。
- **前缀二次迁移风险**：重复加 `BH`、`BT` 等会改变公共身份。迁移前运行 dry-run 并确认 `moved=0`、`deduplicated=0` 才可视为幂等。
- **失败项误接纳风险**：无声、无词法和不支持字符不是同一问题。恢复必须分组抽样，不能统一加 `--force` 后直接进入 accepted。

## 执行检查表

- [ ] 阅读本交接全文并确认仓库、运行根目录和 NAS 均可访问
- [ ] 保存新的 HEAD、分支、工作树状态和状态哈希
- [ ] 确认 77/77 块 complete，最终 counts 守恒
- [ ] 确认 92 个 terminal 事件对应 77 个唯一块
- [ ] 验证 77 份 GAMESL 与 output publication receipt 可解析
- [ ] 复核 speaker migration 的 184,990、78,857、0
- [ ] 复核 393,757 对以及 440 条已解释额外 WAV
- [ ] 验证 `/mnt/Raw/0917/manifest.json` 和 10 组样本
- [ ] 运行针对性测试并保存结果
- [ ] 确认没有修改 v5_0707
- [ ] 在任何写操作前取得数据所有者对开放问题的决定
- [ ] 局部写操作前生成路径、哈希和 rollback 清单

## 就绪判定

- 精确 Sol-high 规划路由：通过。
- 仓库与工作树识别：通过。
- 代码、配置、运行记录和输出证据定位：通过。
- Facts、Assumptions、Decisions、Open Questions 分离：通过。
- R1 至 R12 均有可观察验收标准和验证步骤：通过。
- 新鲜度检查：通过；UTC `2026-09-17T11:27:52Z` 时 77/77、counts 和 0917 manifest 仍匹配。
- 只读接手：就绪。
- 删除 440 个 WAV、回收失败项或提交代码：需要数据所有者/仓库维护者分别决策，当前不属于已授权动作。

## 新会话启动指令

在新会话中先发送以下指令：

> 请在 `/mnt/local_E/MFA_Pause/repo` 接手 Qwen3 0915ALL 全语料项目。第一步完整读取 `/mnt/local_E/MFA_Pause/repo/handoffs/20260917T112752Z-qwen3-0915all-operational-handoff.md`，然后仅执行其中的只读快照与验收步骤。保护当前脏工作树；核对 `/mnt/nvme3/qwen3_0915all_full_20260914`、`/mnt/Raw/0915ALL`、`/mnt/Raw/GAMESL` 和 `/mnt/Raw/0917/manifest.json`。除非证据明确不一致，不重跑全量、不删除 440 条额外 WAV、不修改 v5。发现问题时按失败类别、具体块和 publication receipt 提出最小修复，并保留 rollback 与哈希记录。
