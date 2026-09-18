# 合并后处理步骤与重复数据处理的提效实施交接

## 元数据

- UTC 时间：2026-09-08T07:35:35Z
- 仓库：`/mnt/local_E/MFA_Pause/repo`
- revision：`38c6d740fc318bf25d2bcdf21bedbe7dda6bbe4d`
- branch：`main`
- 工作树：dirty，共 31 项 tracked/untracked 变更
- task slug：`consolidate-postprocessing-work`
- 交接路径：`handoffs/20260908T073535Z-consolidate-postprocessing-work.md`
- 规划路由：根代理已将只读分析派发给 `gpt-5.6-sol`、`reasoning_effort: high`
- 仓库指令：未发现 `AGENTS.md`；已读取 `CLAUDE.md`
- 基线验证：目标后处理回归测试共 351 项通过，未修改实现文件

## 目标

在不改变 TextGrid 几何、tier 内容、过滤结果、证据哈希、失败关闭语义和处理顺序的前提下，合并后处理中的重复全批验证、重复文件读取、重复 JSON 解析、重复派生 tier 重建、重复音频帧计算和嵌套区间扫描。

优先消除随批量规模呈二次增长的生命周期与 strict-English 验证，再处理单 stem 内的重复重建和声学计算。不得通过关闭审计、减少证据校验、改变阈值或提高 worker 数量制造表面提速。

## 背景与当前行为

`step_postprocess` 先构造语料、音频和 aligned TextGrid 集合，校验完整性后启动 `postprocess_textgrids.py`，结束后再次扫描 output、filtered 和报告以验证分区守恒（`scripts/run_pipeline.py:4202-4281`, `scripts/run_pipeline.py:4379-4489`）。

后处理子进程先加载 axis contract、参考文本/WAV 索引和词典，再按 stem 调用 `process_one`；Linux 使用 fork 进程池，Windows 使用线程池，自动 worker 上限为 32（`scripts/postprocess_textgrids.py:19000-19223`）。

`process_one` 分为顺序敏感阶段：声学预处理、文本/tier 初建、CTC/能量/标点边界处理、英文音素注入、后边界处理及最终 QC（`scripts/postprocess_textgrids.py:16148-16174`, `scripts/postprocess_textgrids.py:16487-16633`, `scripts/postprocess_textgrids.py:16673-16857`）。最终 words 几何在 `_freeze_processed_geometry` 冻结，然后 `_rebuild_derived_from_frozen_words` 从冻结 words 事务性重建 publication tiers（`scripts/postprocess_textgrids.py:1064-1104`, `scripts/postprocess_textgrids.py:1107-1199`, `scripts/postprocess_textgrids.py:17657-17672`）。

`finalize_textgrids.py` 是独立 CLI，只对给定目录再做清理；标准 `step_postprocess` 不调用它（`scripts/finalize_textgrids.py:26-43`, `scripts/finalize_textgrids.py:87-124`）。

## Facts

1. `_load_ctc_lifecycle` 在每个 stem 内调用两项全批 validator，然后重新解析和哈希 raw manifest/work receipt（`scripts/postprocess_textgrids.py:936-999`, `scripts/postprocess_textgrids.py:16185-16208`）。
2. `validate_ctc_raw_manifest` 每次遍历并 SHA-256 校验所有 stem 的全部 CTC artifact（`scripts/pipeline_utils.py:848-904`）；`validate_ctc_work_receipt` 每次遍历并哈希所有 work/reference artifact（`scripts/pipeline_utils.py:970-1014`）。N 个 stem 因而可触发约 N 次全批文件验证。
3. `_nvasr_build_producer_authority` 每个 stem 都线性扫描 raw/work receipt 的完整 `files` 列表，并分别读取 raw/work token sidecar（`scripts/postprocess_textgrids.py:796-824`）。
4. 同一 work token JSONL 在 `process_one` 中还会于初始 semantic 投影、CTC snap、NVV 恢复和 swallowed-punctuation timeline 再次解析（`scripts/postprocess_textgrids.py:16350-16363`, `scripts/postprocess_textgrids.py:16530-16541`, `scripts/postprocess_textgrids.py:16862-16874`, `scripts/postprocess_textgrids.py:17256-17269`）。
5. `.txt`/`.lab` 内容在路径选择、fallback、pinyin 和 SHA-256 provenance 构建中重复读取（`scripts/postprocess_textgrids.py:16213-16219`, `scripts/postprocess_textgrids.py:16306-16338`）。
6. strict English 模式下，每个 stem 都重新读取全局 `en_alignment_manifest.json`，扫描完整 `stem_ledgers` 列表寻找本 stem，再读取并哈希本 stem ledger（`scripts/postprocess_textgrids.py:15294-15317`, `scripts/postprocess_textgrids.py:15356-15381`），形成另一条随批量呈二次增长的路径。
7. `build_pinyin_phones_tier` 对每个 word 遍历整个 `pinyin_dict` 做大小写不敏感查找（`scripts/postprocess_textgrids.py:4250-4275`, `scripts/postprocess_textgrids.py:4314-4322`）；该函数又在一个 stem 内被多次调用。
8. Phase 4 先在 D5 条件分支重建 derived tiers，随后无条件重建，E/F/G 后再次无条件重建（`scripts/postprocess_textgrids.py:16731-16749`, `scripts/postprocess_textgrids.py:16769-16776`, `scripts/postprocess_textgrids.py:16844-16851`）。
9. Phase 5 还有多处独立 hanzi/pinyin_phones 重建，最终 frozen publication transaction 又会覆盖性重建（`scripts/postprocess_textgrids.py:17010-17046`, `scripts/postprocess_textgrids.py:17316-17340`, `scripts/postprocess_textgrids.py:17455-17489`, `scripts/postprocess_textgrids.py:17660-17672`）。
10. `_sync_derived_tiers` 同时承担 source-phone lineage 修复、words geometry reconciliation、hanzi rebuild 和 pinyin_phones rebuild，使只需更新 source phones 的路径也执行全部 derived 工作（`scripts/postprocess_textgrids.py:5889-5985`）。
11. 全音频 RMS 至少在 visual silence、ellipsis merge、ellipsis extension、energy refinement、word-energy noise fallback 和 BGM 检测中重复计算（`scripts/postprocess_textgrids.py:9722-9729`, `scripts/postprocess_textgrids.py:12301-12304`, `scripts/postprocess_textgrids.py:12439-12442`, `scripts/postprocess_textgrids.py:12526-12535`, `scripts/postprocess_textgrids.py:9114-9144`, `scripts/postprocess_textgrids.py:17991-18005`）。
12. `_rms_frames_in_span` 为每个 span 重新构造 Python slice 列表和 NumPy frame array，即使相同规格的全局对齐 frame 可从同一 frame bank 选取（`scripts/postprocess_textgrids.py:9068-9096`）。
13. 最终 pinyin-phone containment QC 对每个 phone 使用 `any` 扫描所有 word ranges，为 O(P×W)（`scripts/postprocess_textgrids.py:17941-17959`）。
14. 主流程把所有 futures 和完整 report dict 留在内存，全部结束后才写 JSONL（`scripts/postprocess_textgrids.py:19159-19228`）。
15. 现有代码已证明批次级索引可安全消除递归查找：reference 和 WAV 各自只扫描一次，并有确定性/文件消失 fallback 测试（`scripts/postprocess_textgrids.py:11127-11187`, `scripts/postprocess_textgrids.py:19114-19120`, `tests/test_postprocess_wav_index.py:8-50`）。
16. README 声明后处理输出包括五层 TextGrid、tone mapping 和 report，并记录 words 是边界 authority、其余 publication tiers 从冻结 words 单向重建（`README.md:313-350`, `README.md:569-586`, `README.md:636-639`）。
17. 当前后处理默认开启 strict、短词修复和 BGM，但 word-in-silence filter 默认关闭；生产配置覆盖 authority/no-reference、BGM 开/关组合（`scripts/run_pipeline.py:302-328`, `configs/gamedata_reverse1999_noref_20260903.yaml:63-76`, `configs/gamedata_genshin_reference_20260903.yaml:73-82`）。
18. 当前 dirty 变更包含 `scripts/pipeline_utils.py` 和 `scripts/run_pipeline.py`，但不包含 `scripts/postprocess_textgrids.py`、`scripts/audio_energy.py` 及目标后处理测试；实施必须保留并适配现有修改。

## Assumptions

- 优化目标是大批量吞吐和峰值内存，而非单文件最低延迟。
- 输出等价包括 output/filtered stem 分区、TextGrid 序列化字节、report 语义字段、tone mapping、错误状态和 exit code。
- raw CTC namespace 和 receipt 是只读证据，但仍需检测运行期间替换或篡改。
- 实施窗口可以建立独立 baseline worktree，并获得至少一个 authority 和一个 no-reference 隔离样本批次。

## Decisions

1. P0 先合并生命周期与 strict-English 全批验证，因为它们具有明确的 O(N²) I/O/扫描证据。
2. 批次预检只验证全局结构一次；每个 stem 仍对实际消费的 artifact bytes 计算并核对 receipt hash，结束前再次核对 manifest/receipt 身份，不以缓存削弱失败关闭。
3. 同一 token 文件只解析一次，但保留两个内存视图：未改写 raw view 供 snap/owner evidence 使用，canonical semantic view 供 authority/fallback correspondence 使用。
4. words 保持唯一可变几何 authority；source-phone lineage 同步与 hanzi/pinyin_phones publication rebuild 分拆。只有真实 consumer barrier 才重建，freeze 后只执行一次最终 publication transaction。
5. 声学缓存保留现有 frame size、采样取整、全局对齐和局部对齐差异；不得用 `audio_energy.word_rms` 的 mean-absolute-amplitude 语义替换当前 10ms frame RMS。
6. worker 自动值、过滤阈值、CTC owner 优先级、strict provenance 时序和 report schema 本轮不改变。
7. report 流式化和 O(P×W) QC 优化排在核心等价性工作之后并独立提交，便于回滚。

## Open Questions

| 问题 | 证据与影响 | Owner | 决策路径 |
|---|---|---|---|
| 用哪个隔离批次作为 rollout 基准？ | 仓库有配置但语料位于外部路径；没有同源批次不能判断真实 wall-time 改善 | 数据 owner | 提供固定 stem manifest、只读输入和两个全新输出根；至少覆盖 authority 与 no-reference |
| 是否要求检测任意未消费 CTC artifact 的运行中变化？ | 当前每 stem 重扫全批偶然提供高成本持续复验；新设计默认预检全量、逐 stem 核验消费 bytes、结束复核 receipt | 安全/数据 owner | 若必须持续核验全部文件，采用独立低频完整复核或打开文件描述符快照，不恢复每 stem 全批扫描 |
| report 行顺序是否有未记录的外部依赖？ | 并行路径当前按 future 完成顺序追加，本身不稳定；pipeline 只按 stem set/唯一性验证 | 发布 owner | 搜索下游消费者并加入契约测试；若有顺序要求，明确固定为 input stem 顺序 |

## 范围与约束

### In scope

- 批次级 CTC lifecycle/strict-English preflight 与索引。
- 每 stem 文本、token、punctuation、reference 和 provenance bytes 的单次加载。
- pinyin 字典大小写索引。
- source-phone 同步与 derived publication rebuild 解耦。
- stem 内 RMS/frame cache。
- 最终 phone/word containment 线性扫描。
- 有界任务提交和原子 report 流式写入。
- operation count、输出等价、篡改失败和并行模式测试。
- README 中后处理数据流与性能不变量说明。

### Out of scope

- 修改 CTC/MFA/English MFA 模型、beam、阈值和 owner 规则。
- 调整 worker 数、CPU affinity、批大小或生产配置。
- 删除 strict audit、axis contract、artifact hash 或 denominator 守恒。
- 改写 `finalize_textgrids.py`。
- 修改生产语料、现有输出、历史 handoff 或字典。
- 将重构扩展到其他 pipeline 阶段。

### Constraints

- 保留当前 31 项 dirty 工作；不得覆盖 `scripts/pipeline_utils.py`、`scripts/run_pipeline.py` 的用户修改。
- `CLAUDE.md` 要求：若实施中发现并修复逻辑冲突、误判、kind 标记错配或 phase ordering bug，必须追加 `REGRESSION_ARCHIVE.md`；纯性能等价重构不触发归档。
- raw/canonical token 视图、reference/fallback authority 分离、CTC raw/work 双向证据和 final freeze 顺序不可合并为含混视图。
- Windows 线程池共享对象必须只读；Linux fork 路径不得因 initializer 序列化产生每 worker 大副本。

## 受影响文件与符号

| 文件与证据 | 计划 |
|---|---|
| `scripts/postprocess_textgrids.py:936` `_load_ctc_lifecycle` | 拆成一次 batch preflight 与 per-stem evidence binding |
| `scripts/postprocess_textgrids.py:796` `_nvasr_build_producer_authority` | 接收按 `(stem, suffix)` 建好的 manifest row index 和已解析 token rows |
| `scripts/postprocess_textgrids.py:15294` `load_strict_en_provenance` | 增加可选 batch manifest/index；保留无 context 兼容入口 |
| `scripts/postprocess_textgrids.py:16148` `process_one` | 接收 immutable batch context；统一 stem artifact 加载与 dirty derived state |
| `scripts/postprocess_textgrids.py:19000` `main`、`_worker_init` | worker 启动前 fail-fast preflight，并共享只读 context |
| `scripts/postprocess_textgrids.py:4250` `build_pinyin_phones_tier` | 使用预计算 lowercase-first dictionary index，去掉逐 word 全字典扫描 |
| `scripts/postprocess_textgrids.py:5889` `_sync_derived_tiers` | 分拆 lineage/source-phone barrier 与 display-tier rebuild |
| `scripts/postprocess_textgrids.py:1107` `_rebuild_derived_from_frozen_words` | 保持 freeze 后唯一完整 publication transaction |
| `scripts/postprocess_textgrids.py:9055` `_frame_rms_vec`、`scripts/postprocess_textgrids.py:9068` `_rms_frames_in_span` | 接入共享 frame cache，保留数值语义 |
| `scripts/audio_energy.py:30` `frame_rms`、`scripts/audio_energy.py:73` noise helpers | 增加区分 global/local alignment 的可复用缓存对象；不改变旧 API |
| `scripts/postprocess_textgrids.py:17941` phone containment QC | 用有序双指针替代每 phone 扫描全部 words |
| `scripts/postprocess_textgrids.py:19159` report aggregation | 有界提交 futures；结果写临时 JSONL，成功后原子替换 |
| `scripts/run_pipeline.py:4436` report contract reader | 逐行读取 report，避免 `read_text().splitlines()` 全量物化；先合并现有 dirty diff |
| `tests/test_postprocess_efficiency.py` | 新增 batch validation/read/rebuild/frame operation-count 与输出等价测试 |
| `tests/test_ctc_artifact_versions.py:858` | 扩展 lifecycle tamper、freeze 和 derived transaction 覆盖 |
| `tests/test_postprocess_word_energy.py:232` | 扩展缓存前后数值与 frame-alignment 等价覆盖 |
| `tests/test_hyphenated_postprocess_provenance.py:17` | 验证 strict manifest index 不改变 English binding |
| `tests/test_postprocess_wav_index.py:8` | 复用既有确定性索引/fallback 模式 |
| `README.md:541`, `README.md:636` | 记录一次 batch preflight、单次 stem 读取和 freeze barrier |
| `scripts/pipeline_utils.py:848`, `scripts/pipeline_utils.py:970` | 只作为 validator 依赖；除非可无冲突增加“已解析 payload”参数，否则不修改该 dirty 文件 |

## Numbered Requirements

1. CTC raw manifest 和 work receipt 的全批 schema、membership、size/hash 验证每次 postprocess invocation 各执行一次，不再每 stem 执行。
2. 每 stem 的 work token JSONL、raw token JSONL、punctuation JSON、选定 transcript 和 reference bytes 各最多读取/解析一次；消费 bytes 必须与 batch receipt index 的 hash/size 一致。
3. strict-English 全局 manifest 每次 invocation 解析和验证一次，按 stem O(1) 定位 ledger；每个需要的 ledger 仍独立核验 path/hash/schema。
4. pinyin pronunciation lookup 使用一次构建的 lowercase-first index，并保持原来“字典插入顺序中第一个大小写匹配项获胜”的行为。
5. words/source phones/derived tiers 使用显式 dirty/barrier 模型；无 consumer 的中间 hanzi/pinyin_phones rebuild 被移除，freeze 后仅有一次完整 publication rebuild。
6. 一个 stem 内相同 frame specification 的全音频 RMS 只计算一次；global-aligned span、local segment、percentile 和阈值结果与当前实现数值等价。
7. phone-to-word containment QC 从 O(P×W) 降为 O(P+W)，issue 数量、原因和 report 字段保持一致。
8. 并行调度只保留有界数量未完成 futures，report 逐结果写入同目录临时文件；只有所有 worker 和结束证据复核成功才替换正式 report。
9. authority/fallback、strict/non-strict、BGM 开/关、含/不含 English/NVV 的基线与候选输出必须语义等价；候选只有在代表批次三次运行中位 wall time 降低且 peak RSS 不恶化超过 5% 时才可 rollout。

## 有序实施计划、依赖与 Ownership

1. **性能 owner：建立基线**
   - 从当前 revision 和 dirty 内容建立只读 baseline 快照或隔离 worktree。
   - 固定 authority/no-reference 样本 stem、输入 hashes、配置和预期 output/filtered/report。
   - 依赖：数据 owner 提供隔离语料；不改实现。
2. **证据 owner：实现 batch lifecycle context**
   - 在 `main` 中于 worker 启动和任何 candidate 输出前解析/验证 raw manifest、work receipt 各一次。
   - 构造 stems set、`(stem, suffix)` row indexes、manifest hashes/stats 和公共 lifecycle report projection。
   - 将 context 作为只读 initializer 数据传入线程/进程 worker。
   - 依赖：步骤 1 基线。
3. **证据 owner：实现 per-stem artifact bundle**
   - 单次读取 bytes 后同时完成 hash、decode 和 JSON parse。
   - 从一份 work token rows 派生 raw-use view 与 canonical semantic copy。
   - `_nvasr_build_producer_authority` 使用已加载 raw/work rows 和 row index。
   - 结束前复核 manifest/receipt identity；任何不一致令 invocation 非零退出且不发布完整 report。
   - 依赖：步骤 2。
4. **English owner：索引 strict provenance**
   - preflight 全局 manifest partition/schema 一次，按 stem 建 ledger entry index。
   - `load_strict_en_provenance` 增加可选 context，保留现有直接调用行为供测试和工具使用。
   - 依赖：步骤 2；可与步骤 3 同提交但单独测试。
5. **tier owner：加入 pinyin lookup index**
   - 从 `load_dict` 结果以 `setdefault(key.casefold(), phones)` 构造 lookup。
   - 所有 `build_pinyin_phones_tier` 路径使用 O(1) lookup。
   - 依赖：无；独立提交。
6. **tier owner：分拆同步 barrier**
   - 从 `_sync_derived_tiers` 提取 source-phone lineage reconcile。
   - 每个 words mutation 返回 changed flag 或增加 revision；只有后续 consumer 需要时同步。
   - E/F/G 只提交 words owner 结果；被 frozen rebuild 覆盖的中间 pinyin_phones 不再构建。
   - `_restore_reference_surfaces` 先恢复 words surface，hanzi 由 final transaction 生成。
   - freeze 前确保 visual resolver 所需 source phones 当前；freeze 后完整 rebuild 仅一次。
   - 依赖：步骤 1 golden outputs。
7. **声学 owner：实现 per-stem frame cache**
   - `audio_energy.py` 增加以 audio identity、sample rate、frame size、alignment mode 为 key 的缓存。
   - 将 refine、visual owner、ellipsis、word-energy 和 BGM 的 full-audio frame 请求接入同一 context。
   - span 选择复用 global 10ms bank；局部 segment 保留独立局部起点。
   - 依赖：步骤 6 稳定后执行。
8. **QC/report owner：线性扫描与有界结果流**
   - containment 使用有序双指针。
   - executor 只维持不超过 `max(2 × workers, 4)` 个未完成任务。
   - report 写同目录临时文件，聚合仅保留 counts、hard-integrity count 和必要错误摘要。
   - `run_pipeline.py` report contract 改为逐行读取。
   - 依赖：步骤 2 batch end verification；实施前手工合并该文件现有 dirty 变更。
9. **质量 owner：回归与基准**
   - 执行目标测试、baseline/candidate artifact comparison、tamper cases 和三次代表批次性能测量。
   - 只有 R1-R9 全部通过才进入 rollout。

## Requirement-to-Acceptance Traceability

| Requirement | Observable acceptance | Verification |
|---|---|---|
| R1 | 1、10、100 stem fixture 中两个全批 validator 调用数恒为各 1；总 artifact hash 次数随 N 线性增长 | `tests/test_postprocess_efficiency.py` monkeypatch counters |
| R2 | 每 stem 每类 artifact read/decode 不超过 1；任一已读 bytes 与 receipt 不符时无正式成功 report | 单次读取计数与 mid-run replacement 测试 |
| R3 | strict manifest read/parse 恒为 1；stem lookup 不遍历完整 ledger list；ledger hash 每所需 stem 1 次 | strict 100-stem operation-count 测试及 hyphenated provenance tests |
| R4 | lookup 构造一次；混合大小写和重复 case key返回与当前首次匹配相同 phones | 参数化 dictionary compatibility tests |
| R5 | words 未变化时 rebuild count 不增加；visual commit 到 freeze 之间无完整 derived rebuild；最终 TextGrid bytes 与 baseline 相同 | rebuild spy、processed geometry digest 和 golden artifact comparison |
| R6 | 同一 full-audio 5/10ms bank 各计算最多一次；所有 RMS arrays、noise floor、classification、BGM decisions 与 baseline 相同 | word-energy、visual-owner、ellipsis 数值测试 |
| R7 | operation counter 不超过 P+W 常数倍；issues/report 与旧算法 golden 相同 | 随机有序 interval differential test |
| R8 | pending futures 不超过设定窗口；worker error/tamper 时正式 report 不替换；成功 report stem set 完整唯一 | executor/report fault-injection tests |
| R9 | output/filtered 集合、TextGrid bytes、tone mapping、report 语义和 exit code 无差异；三次中位 wall time 下降且 peak RSS 增幅不超过 5% | 双 worktree 比较、严格审计、`/usr/bin/time -v` 汇总 |

## 可观察验收标准

1. 合成 N-stem lifecycle 测试从全批验证调用数 `2N` 降至 2，总 per-stem artifact 核验保持线性。
2. 100-stem strict manifest fixture 只读取一次全局 manifest，并准确拒绝 duplicate/missing/rejected ledger partition。
3. raw token view 的字节、顺序和字段不变；authority canonicalization 只发生在 semantic copy。
4. baseline 与候选每个 stem 的最终 TextGrid SHA-256 相同。
5. baseline 与候选 output/filtered/report stem set、filter reasons、hard-integrity reasons、processed geometry digest 和 provenance hashes 相同。
6. `word_energy_audit` 的 noise model、阈值、每词 classification 与 baseline 相同。
7. 缓存后同一 frame bank 不重复构造；非标准 sample rate 和非整数 frame size 与旧函数相同。
8. 任一 manifest、receipt 或所消费 sidecar 在 preflight 后被替换时，进程非零退出且不发布看似完整的新 report。
9. 并行路径未完成任务和内存中完整 report 数量受 worker 窗口约束。
10. 代表批次三次运行的候选中位 wall time 低于 baseline，peak RSS 不超过 baseline 的 105%。

## 验证命令与预期信号

```bash
cd /mnt/local_E/MFA_Pause/repo
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_postprocess_efficiency.py \
  tests/test_postprocess_wav_index.py \
  tests/test_postprocess_word_energy.py \
  tests/test_postprocess_recovery_geometry.py \
  tests/test_hyphenated_postprocess_provenance.py
```

预期：全部通过；validator、artifact read、manifest parse、derived rebuild 和 frame-bank counters 满足 R1-R6。

```bash
cd /mnt/local_E/MFA_Pause/repo
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_postprocess_geometry.py \
  tests/test_ctc_artifact_versions.py \
  tests/test_axis_contracts.py \
  tests/test_no_reference_mode_compat.py \
  tests/test_boundary_punctuation_display_regressions.py \
  tests/test_authority_publication_contract.py \
  tests/test_laria_bgm_ctc_gap.py
```

预期：全部通过；当前基线分别为第一组现有测试 49 项、第二组 302 项，共 351 项通过。新增效率测试在实现后计入第一组。

```bash
cd /mnt/local_E/MFA_Pause/repo
PYTHONDONTWRITEBYTECODE=1 python -m compileall -q scripts/postprocess_textgrids.py scripts/audio_energy.py
```

预期：退出码 0，不生成仓库内字节码。

```bash
cd /mnt/local_E/MFA_Pause/repo
PYTHONDONTWRITEBYTECODE=1 python tests/benchmark_postprocess.py \
  --synthetic-stems 100 --repeat 5 \
  --json-out /tmp/postprocess-efficiency-100.json
```

预期：JSON 记录 validator/read/frame/rebuild counts、wall time 和 peak RSS；计数满足线性边界。该 benchmark 文件由步骤 1 随效率测试夹具实现，不进入默认生产路径。

代表语料必须在两个隔离工作区运行 baseline 和 candidate，各三次，以 `sha256sum` 比较 TextGrid/tone mapping，并逐行规范化仅含绝对输出路径的 report 字段后比较 JSON。任何 stem set、filter reason、geometry digest 或 provenance hash 差异均判定失败。

## Cautions 与 Invariants

- Phase A snap、B energy refine、C punctuation inject、Phase 3.5 English injection、Phase 4 owner mutation 和最终 freeze 的语义顺序不可改变。
- `_nvasr_build_producer_authority` 必须继续比较 sealed raw rows 与 work rows，不得只信 receipt 元数据。
- raw token rows 与 canonical semantic rows 不可共享可变 dict。
- visual short-silence resolver 读取 source-phone lineage；删除 derived rebuild 前必须保留明确的 source-phone barrier。
- strict English phones 只能在最终几何稳定后注入，通用 de-overlap 不得改写 strict `en:` geometry。
- frame cache 必须区分全局整帧对齐和切片局部对齐，否则边界附近分类会变化。
- `audio_energy.word_rms` 返回 mean absolute amplitude，不能替代 postprocess 的 10ms-frame RMS。
- report 行顺序不是当前并行契约，但 stem 唯一性、完整集合和字段内容是契约。
- output 与 filtered 必须互斥且并集等于 eligible denominator。
- 不得因性能重构吞掉 OSError、JSON decode 错误、hash mismatch 或 worker exception。

## 风险与回滚

| 风险 | 缓解 | 回滚 |
|---|---|---|
| batch cache 产生 TOCTOU 窗口 | 对消费 bytes 即时 hash；结束复核 manifest/receipt；正式 report 原子发布 | 回退 R1-R3 提交，恢复 per-stem loader |
| fork/线程共享 context 被意外修改 | frozen dataclass、tuple/mapping proxy、worker 只读测试 | 禁用共享 context，保留 batch preflight 结果序列化副本 |
| derived rebuild 减少后读到 stale tier | revision/dirty 断言；consumer barrier 测试；final digest 比较 | 回退 R5 提交，不影响证据缓存 |
| RMS cache 改变 frame 取整 | global/local alignment 分 key；逐数组 differential tests | 回退 R6 提交，继续保留 R1-R5 |
| report 流式化留下部分文件 | 同目录临时文件、fsync/close 后 atomic replace、失败清理临时文件 | 回退 R8 提交 |
| `run_pipeline.py` dirty 冲突 | 实施前保存 diff，只改 report 读取局部 | 跳过 parent-side streaming；核心提效不依赖该项 |
| 性能改善不足 | operation count 仍作为正确性收益，但不 rollout 低收益 R5-R8 | 按独立提交逐项回退 |

无需数据迁移。回滚以独立提交为单位，不删除 baseline/benchmark 证据。

## Blockers 与 Questions

| 项目 | 证据 | 影响 | Owner | Decision route |
|---|---|---|---|---|
| 代表语料路径未确定 | 仓库配置引用外部数据，仓库内无可直接运行的完整批次 | 不阻塞 R1-R8，阻塞 R9 生产 rollout | 数据 owner | 提供固定 stem manifest 和隔离输出根 |
| `run_pipeline.py`、`pipeline_utils.py` 已有用户改动 | 当前 git status 显示二者 modified | 阻塞重叠 hunk 的直接覆盖 | 实施者 | 先保存并审阅现有 diff；优先不改 pipeline_utils，只手工合并 run_pipeline report reader |
| 全批持续篡改检测强度待确认 | 当前高成本 per-stem 重验会重复检查未消费 artifact | 影响 batch cache 安全边界 | 安全/数据 owner | R2 前确认“预检全量+消费时 hash+结束 receipt 复核”是否满足要求 |
| report 顺序外部依赖未知 | 当前并行输出顺序不稳定，仓库内校验只使用集合 | 影响 R8 是否固定排序 | 发布 owner | 搜索外部消费者；无证据则保持 completion-order 语义 |

## 执行检查清单

- [ ] 保存当前 revision、status 和 dirty diff 快照
- [ ] 建立 baseline worktree 和固定样本 manifest
- [ ] 增加 operation-count 与 golden-output 测试
- [ ] 实现并验证 batch lifecycle context
- [ ] 实现 single-read per-stem artifact bundle
- [ ] 实现 strict manifest stem index
- [ ] 实现 lowercase-first pinyin lookup
- [ ] 分拆 source-phone 与 derived publication barriers
- [ ] 接入 global/local-aware frame cache
- [ ] 将 containment QC 改为双指针
- [ ] 实现有界 futures 与 atomic report
- [ ] 运行 351 项当前目标回归及新增测试
- [ ] 执行 authority/no-reference 双 worktree 等价比较
- [ ] 执行三次代表批次基准并审阅 RSS
- [ ] 若包含逻辑行为修复，按 `CLAUDE.md` 追加 `REGRESSION_ARCHIVE.md`
- [ ] 记录每个独立提交的回滚点

## Readiness 与 Gates

- Sol-high 精确路由：PASS。
- 仓库、revision、branch、dirty 状态：PASS。
- 仓库指令读取：PASS。
- entrypoint、call flow、数据流、测试和配置证据：PASS。
- Facts/Assumptions/Decisions/Open Questions 分离：PASS。
- requirements 与 acceptance/verification 映射：PASS。
- 当前目标测试基线：PASS，351 项通过。
- handoff 候选路径冲突检查：PASS，路径空闲。
- freshness：PASS；最终检查时 revision 为 `38c6d740fc318bf25d2bcdf21bedbe7dda6bbe4d`，branch 为 `main`，dirty 项为 31。
- 仓库修改：PASS；规划阶段未编辑实现文件。
- 实施准备度：READY。
- 生产 rollout：BLOCKED，等待代表语料、篡改检测强度和 report 顺序问题完成决策，并通过 R9。

## Next-window startup instructions

在 `/mnt/local_E/MFA_Pause/repo` 开始实施窗口，先读取 `/mnt/local_E/MFA_Pause/repo/handoffs/20260908T073535Z-consolidate-postprocessing-work.md`。立即重新记录 `git rev-parse HEAD`、`git status --short`，保存 `scripts/run_pipeline.py` 与 `scripts/pipeline_utils.py` 的现有 diff，并从当前 dirty 内容建立隔离实现环境。

第一项代码工作只实现 `tests/test_postprocess_efficiency.py` 的全批 validator/read counters 和 baseline artifact golden checks；随后完成 R1 batch lifecycle context。R1-R4 通过后再进入 derived-tier 与 audio cache 重构，不得在同一提交中混合证据缓存、tier barrier 和声学数值变化。
