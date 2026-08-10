# 主线全模式逻辑完整性实施交接方案

## 元数据

- 生成时间：2026-08-07T09:15:14Z
- 仓库：/mnt/local_E/MFA_Pause/repo
- 分支：main
- 基线 revision：07a6fa2195ccb3aa9019c9bddaad3eb6afac02ce
- 基线状态：审查快照包含未提交改动：scripts/ctc_prealign.py、scripts/run_pipeline.py、scripts/streaming_pipeline.py，以及新增 configs/hecheng_ria_test1.yaml；这些改动属于既有工作，必须保留。
- 任务 slug：mainline-all-modes-logic
- 本交接文件：handoffs/20260807T091514Z-mainline-all-modes-logic.md
- 规划路由：已由 gpt-5.6-sol、高推理强度的只读规划工作者完成审查。

## 目标与具体任务

修复并验证主管线及其批处理/流式适配层，使 full、ctc_ready、batch_ctc_ready、nvrasr_fallback 四种模式对每个输入 stem 都有唯一、可追溯的终态：成功、过滤或失败；不得因陈旧文件、部分产物、复制失败、子进程异常或发布失败而静默缩小分母或返回成功。

本文件只规划后续实现，不实现代码、不重写测试、不修改配置、不回退当前工作树。

## 背景与当前行为

主入口在 scripts/run_pipeline.py:3376-3435 接受四种模式。路由表位于 scripts/run_pipeline.py:3316-3337：

- full：trim → resample → prealign → normalize_punct → normalize → normalize_ria → normalize_en → adjust → align → align_en → postprocess → strict_ok。
- 普通 ctc_ready：link → pad_silence → normalize_punct → normalize → normalize_ria → normalize_en → resample → adjust → align → align_en → postprocess → strict_ok。
- 严格 evidence ctc_ready：同一条输入链，但去除 pad_silence。
- nvrasr_fallback：prealign → pad_silence → normalize_punct → normalize → normalize_ria → normalize_en → resample → adjust → align → align_en → postprocess → strict_ok。

REGRESSION_ARCHIVE.md:125-131 仍只把三种单数据集模式列为架构说明，README.md:49-57,160-173 仍按旧的八步管线描述；文档与可执行入口已经漂移。

当前 configs/hecheng_ria_fresh.yaml:38-47 开启 ctc_prealign.all_gpus: true。主入口在 scripts/run_pipeline.py:826-852 转发 --all-gpus，但该路径当前不能视为已闭环。

## 事实（Facts）

1. 当前工作树的 scripts/ctc_prealign.py:1363-1498 在合并逻辑中打开 try 后，后续合并语句没有保持在该 try 的缩进块内；对当前文件执行 AST 解析在 1395 行报告 SyntaxError: expected 'except' or 'finally' block。因此当前工作树不能作为可执行基线。
2. all-GPU preflight 在 scripts/ctc_prealign.py:1274-1293 允许的 namespace 没有包含 .ctc_run_receipt.json，但同一路径在 scripts/ctc_prealign.py:1329-1348 又强制要求该文件。receipt writer 在 scripts/pipeline_utils.py:1566-1614 明确写入该文件；即使先修复语法，namespace 校验仍会拒绝正常 shard。
3. pipeline_utils.py:1617-1644 还提供 .shard_receipt.json 写入接口，但当前 CTC 编排没有用它；receipt schema、文件名和 preflight 契约需要统一。
4. 当前未提交改动已把单卡 receipt 的输入 stem 转换修正为 Path(p).stem，见当前 scripts/ctc_prealign.py:2131-2147 附近；也已把 MFA shard 的 _failed 与 _return_codes 提前初始化，见当前 scripts/run_pipeline.py:1432-1436。这些既有改动不得覆盖。
5. step_prealign 只要发现一个既有 TextGrid 就在 scripts/run_pipeline.py:810-813 返回成功，没有核验 expected stem、完整 CTC bundle、manifest、marker 或 receipt。
6. step_adjust_ctc 只要发现一个既有 adjusted TextGrid 就在 scripts/run_pipeline.py:1272-1278 复用该目录；step_mfa_align 随后以存在的 .lab 子集建立 alignment 分母，见 scripts/run_pipeline.py:1644-1652,1734-1737。
7. step_link_ctc 对可解析的旧 ctc_ready_manifest.json 在 scripts/run_pipeline.py:2948-2959 直接短路；实际链接阶段的缺失文件仍只是 warning，见 scripts/run_pipeline.py:3180-3202。
8. CLI 允许 --skip-to 把不属于当前模式的步骤追加到 route，见 scripts/run_pipeline.py:3798-3804；full 的 --scan-only 会保留首步 trim，而不是只做只读扫描，见 scripts/run_pipeline.py:3808-3817。--validate 失败只打印、不影响最终失败集合，见 scripts/run_pipeline.py:3911-3919。
9. batch_ctc_ready 自行创建输出目录并执行普通 CTC_READY_STEP_ORDER，见 scripts/run_pipeline.py:3533-3660，没有复用单数据集严格隔离/版本化发布契约；缺失音频数据集还会被 skip 而不进入 fail_list。
10. streaming prefetch 只以缺失音频决定 ok，没有把 CTC 复制失败纳入队列门禁，见 scripts/streaming_pipeline.py:681-740；StreamingPipeline.run 无论 _process_batch 成败都入 upload queue，见 scripts/streaming_pipeline.py:859-893。
11. _run_direct 在 scripts/streaming_pipeline.py:901-919 调用子进程但丢弃 return code；main 在 scripts/streaming_pipeline.py:1150-1156 也没有将 direct 结果作为进程退出状态。
12. 当前未提交改动已让 _upload_one_batch 的非零 rsync 返回失败，见当前 scripts/streaming_pipeline.py:320-350；但 staged CPU 上传仍在 scripts/streaming_pipeline.py:2400-2434 对 rsync/复制失败只告警并返回成功，批处理入口仍在 scripts/streaming_pipeline.py:2136-2150,2437-2692 打印结果而不把失败转成退出码。
13. 回归档案记录 Cases 97–102 的多项代码已实施但仍待 GPU canary，见 REGRESSION_ARCHIVE.md:7430-7457；Case 103 的 all-GPU 转发也记为已实施，见 REGRESSION_ARCHIVE.md:7461-7483。当前语法错误与 all-GPU namespace 矛盾说明“已实施”不能等同于“可执行且已验收”。
14. load_config 仅做递归合并，见 scripts/run_pipeline.py:307-334，没有统一的未知键、类型、路径关系或模式依赖校验。
15. 工作区快照显示当前未提交改动由审查期间外部并发出现；本方案作者没有修改这些实现文件。

## 假设（Assumptions）

- “每个模式完整无错误”定义为：冻结的输入 stem 集在各阶段守恒，output、filtered 和失败集合互斥且并集等于输入；任何无法证明守恒的阶段必须非零失败。
- --force 可以继续收集诊断，但最终退出码仍必须反映任何失败。
- 旧缓存可以复用，但只能通过完整的 manifest、bundle、内容身份和 provenance 校验；无法证明的旧 marker 进入显式迁移或 fresh rerun。
- 生产 GPU、MFA、CIFS/NAS 状态不能由静态代码审查推断，必须由 canary 记录证明。

## 决策（Decisions）

1. 以“冻结分母 + 阶段 manifest + 运行 receipt”为所有模式的共同契约，不只用于严格英文 evidence 路径。
2. 任何“存在一个文件即可跳过”的 fast path 改为 exact-set 校验；不合格复用必须拒绝或要求新 workspace。
3. CTC、MFA、postprocess、batch、streaming 和 publish 全部采用 fail-closed 退出语义；传输错误不是 warning。
4. output、filtered、audit manifest、publish target 统一使用 run-specific 路径，并由机器可读的结果 receipt 传递真实路径。
5. 保留当前所有未提交改动；后续实现不得使用 reset、checkout 或覆盖写入清除它们。

## 未决问题（Open Questions）

| 问题 | 证据与影响 | 负责人 | 决策路径 |
|---|---|---|---|
| v3 legacy marker 是否继续支持 | scripts/run_pipeline.py:855-895 仍接受缺少内容身份的旧 marker；继续接受会保留陈旧 bundle 风险 | 管线维护者 | 决定移除，或定义一次性迁移验证并增加 schema 版本 |
| strict_ok 在 batch/streaming 中的策略 | 单数据集会写 private run，batch 分支直接预建目录，见 scripts/run_pipeline.py:3533-3660,3733-3755 | 运维/产品与维护者 | 统一 strict contract，或显式命名另一种发布策略 |
| --scan-only 语义 | full 当前会执行 trim，见 scripts/run_pipeline.py:3808-3817 | CLI 维护者 | 拒绝 full/fallback，或实现真正只读 inventory |
| Case 85 的 NVV 策略 | REGRESSION_ARCHIVE.md:6599-6627 记录 nvv_enabled:false 与音频发现需求冲突 | 数据/标注负责人 | 明确 reference-only 与 acoustic-NVV 的适用模式 |
| 缺失参考文本的分母 | archive 前置条件要求隔离 036000，严格代码还包含另一类排除，见 REGRESSION_ARCHIVE.md:7451-7457 与 scripts/run_pipeline.py:2263-2268 | 数据权威负责人 | 冻结 canary stem 文件与缺失参考隔离清单 |
| [PAUSE] phonetic mapping | verify_mapping.py 的预期与当前 spn → sil 反向映射不一致 | 音系 schema 负责人 | 决定映射修复或更新规范，并保留回归 |

## 范围与约束

### 范围内

- run_pipeline.py 四种模式的 route、分母、复用、CLI、阶段失败和发布语义。
- ctc_prealign.py 单卡/all-GPU receipt、shard namespace、manifest/marker 原子性，以及当前语法错误。
- run_pipeline.py MFA shard 启动/超时/输出集合检查。
- batch_ctc_ready 与 streaming_pipeline.py 的 stage/process/upload/cleanup/exit contract。
- 代表性配置、README、回归档案和新增无 GPU 编排回归。
- Cases 80–103 的静态状态与真实 canary 关闭条件。

### 范围外

- 本窗口不重跑生产 54,000 条数据，不启动 GPU，不发布 NAS，不删除旧输出。
- 不在没有数据负责人决策时擅自改变 NVV 语义、缺失参考文本处理或 [PAUSE] 音系规范。
- 不改动用户当前未提交文件，除非下一窗口明确将其纳入实现并保留其意图。

## 受影响文件与符号

| 文件 | 符号/区域 | 证据 |
|---|---|---|
| scripts/run_pipeline.py | FULL_STEP_ORDER、CTC_READY_STEP_ORDER、NVASR_FALLBACK_STEP_ORDER、main | scripts/run_pipeline.py:3316-3337,3340-4013 |
| scripts/run_pipeline.py | step_prealign、step_adjust_ctc、step_link_ctc | scripts/run_pipeline.py:803-899,1264-1295,2933-3246 |
| scripts/run_pipeline.py | _run_mfa_sharded、step_mfa_align | scripts/run_pipeline.py:1298-1633,1636-1818 |
| scripts/ctc_prealign.py | main、all-GPU preflight/merge、receipt、offset/limit | scripts/ctc_prealign.py:1111-1502,1530-2155 |
| scripts/pipeline_utils.py | marker、bundle、model digest、receipt writer | scripts/pipeline_utils.py:1024-1148,1513-1644 |
| scripts/streaming_pipeline.py | prefetch/process/upload、_run_direct、batch main | scripts/streaming_pipeline.py:650-919,1106-1156,2375-2692 |
| 配置/文档 | all-GPU 配置、默认配置、模式说明 | configs/hecheng_ria_fresh.yaml:38-47; config.yaml:159-299; README.md:49-57,160-197 |
| 回归证据 | Cases 80–103、执行记录与前置条件 | REGRESSION_ARCHIVE.md:6475-7483 |

## 编号要求

1. R1 分母守恒：每种模式在第一阶段冻结排序、唯一的 expected stem 集，并由每个后续阶段验证同一集合。
2. R2 CTC 事务：修复当前 ctc_prealign.py 语法错误；统一 .ctc_run_receipt.json/.shard_receipt.json 契约；单卡和 all-GPU 仅在最终 bundle、manifest、receipt、summary、marker 全部成功后发布完成态。
3. R3 严格复用：prealign、link、normalize、adjust 和 MFA 不得以单个文件存在作为完成证明；部分、陈旧或内容身份不一致的产物必须拒绝复用。
4. R4 MFA 完整性：Popen 启动错误、超时、信号退出、非零退出、缺失/额外/非法 TextGrid 都必须形成结构化非零失败，且不能合并为成功目录。
5. R5 CLI/模式契约：--step、--skip-to、--scan-only、--stop-after、--force、strict route 和 all-GPU 组合必须在写目录前被明确接受或拒绝。
6. R6 batch/streaming 事务：复制、子进程、审核、上传、checkpoint 和 cleanup 的失败必须进入聚合状态；失败批次不得上传或被误计为成功。
7. R7 输出隔离：四种模式、batch 和 streaming 都必须使用 run-specific output/filtered/audit/publish 路径，并拒绝非空目标的隐式合并。
8. R8 配置校验：增加类型、未知键、必填路径、模式依赖、strict pins、all-GPU、NVV 和 streaming 资源的统一校验。
9. R9 编排回归：增加无 GPU 的四模式 route matrix、故障注入、receipt/manifest、MFA shard、batch、streaming 和退出码测试。
10. R10 证据与文档同步：README、config.yaml、代表性配置、回归档案和验证器的模式、状态、命令、测试数必须与执行行为一致。
11. R11 发布门禁：CPU fixture、GPU canary、MFA shard canary、CIFS/NAS transfer canary 和版本化发布验证全部通过后才能生产运行。

## 有序实施计划

1. **冻结现状并补故障回归。** 先保留当前工作树；修复前用 AST 检查确认语法错误；为单卡 receipt、all-GPU namespace、MFA Popen、partial artifact、batch 缺失数据、streaming copy/child/rsync 失败建立失败测试。
2. **建立共同 run contract。** 在创建 workspace 前解析 mode 和 CLI，生成 immutable route、expected stems、输入来源和输出策略；不兼容组合直接返回非零。
3. **修复 CTC 事务。** 先修复 try/finally 缩进；决定并实现单一 receipt schema；把 receipt 纳入允许 namespace；校验 schema、model/dict digest、输入/输出 stem 集和 artifact 一一对应；父 marker 最后原子写入。
4. **移除陈旧短路。** 将 prealign、link、adjust、normalization marker 和 MFA 的存在性判断替换为 exact manifest/bundle/content identity 校验；无效复用必须 fresh fail 或显式重跑。
5. **加固 MFA。** 统一初始化状态、捕获启动/超时/信号、必要时终止剩余进程；保留日志和失败 manifest；仅在 merged TextGrid 的精确集合和 strict parser 全通过时返回成功。
6. **贯穿分母到后处理。** 将 expected stems 传给 align、align_en、postprocess、strict_ok、filtered 报告和 publish manifest，保证 input = output ∪ filtered ∪ failed 且集合互斥。
7. **统一 batch/streaming 状态机。** prefetch 失败不入 process queue；process 失败不入 upload queue；upload/目标核验失败不 cleanup 成功；所有聚合函数返回 bool/int 并由模块入口 SystemExit 传播。
8. **统一输出路径。** 由 run_pipeline 产生 machine-readable result receipt，包含实际 output、filtered、audit、publish 路径和集合摘要；streaming 读取 receipt，不再猜测 local_dir/output。
9. **加入配置 schema。** 校验字段类型、路径存在性、模式与步骤依赖、ctc_adjust/pad_silence/nvv_enabled 组合、strict evidence、all-GPU 与 streaming 参数。
10. **修复验证器与文档。** 修正 prepare fixture、mapping 规范和 verify_risks.py 的导入/退出码；仅在相应测试通过后更新 archive 状态和 README。
11. **分级 canary 与发布。** 先跑无 GPU fixtures，再跑单卡 CTC、all-GPU、sharded MFA、四种主模式、streaming transfer failure、严格英文和版本化 NAS canary；最后才允许全量运行。

## 要求—验收追踪

| 要求 | 客观验收标准 | 验证步骤 |
|---|---|---|
| R1 | 任一阶段删除、增加或替换 stem 后返回非零；正常运行各阶段 manifest 集合完全相同 | verify_pipeline_modes.py 的集合突变测试；四模式 canary manifest 对比 |
| R2 | 单卡 CTC receipt 可写；all-GPU shard receipt/namespace 可验证；冲突或坏 shard 不产生父 marker/final manifest | verify_ctc_orchestration.py；mock subprocess all-GPU fault cases；GPU canary |
| R3 | 单个 TextGrid、旧 manifest、v3 marker 或 partial adjusted 目录不能使阶段报告成功 | partial raw/adjusted/link/marker fixture |
| R4 | Popen OSError、timeout、非零、缺 TG、坏 TG、额外 TG 均为结构化非零；成功 merged 集等于 expected | MFA shard fault suite；strict TextGrid parser tests |
| R5 | 每个 mode/CLI 组合在写目录前要么形成明确 route，要么非零拒绝；--validate 失败影响最终退出码 | CLI mode matrix；--skip-to、--scan-only、--force fault tests |
| R6 | failed copy/child/audit/rsync 不进入后续 queue、不上传、不计入成功；进程退出码非零 | verify_streaming_failures.py 与 batch aggregation tests |
| R7 | output、filtered、audit、publish 路径包含唯一 run ID；非空目的地被拒绝；streaming 上传路径来自 receipt | single/batch/stream publication fixture |
| R8 | 错误类型、未知键、缺路径、strict/all-GPU/NVV 冲突在执行前非零 | config schema matrix |
| R9 | 新编排测试覆盖四模式、单/all-GPU、MFA shard、batch、direct/staged streaming 和 exit propagation | 测试清单与 CI 输出 |
| R10 | README/config/archive 的模式数、步骤、命令、状态与 --list-steps/验证器输出一致 | 文档一致性检查 |
| R11 | 所有 CPU/GPU/MFA/NAS canary 记录 expected 集合、receipt、strict audit 和发布结果，全部成功才放行 | 冻结 canary stems 文件、运行 receipt、publish manifest 审阅 |

## 可观察验收标准

- 当前工作树首先通过 AST 解析，且不覆盖既有未提交改动。
- 四种 run_pipeline.py 模式各自打印的 route 与 route registry 一致；非法 route 在首次写目录前失败。
- 任一正常阶段的输入/输出集合可由 manifest 和 receipt 重建，集合不缺、不重、不交叉。
- CTC 成功态同时具备合法六件套、最终 manifest、summary、receipt 和 marker；失败态没有可误认的父完成 marker。
- MFA shard 的日志、失败原因、missing/extra/invalid 集均可读，成功合并目录无额外或缺失 stem。
- streaming 的 prefetch、process、upload 三个计数相等且等于总批次；任一错误使最终返回非零并保留证据。
- 发布目标是空的、版本化的、经过独立 strict audit 的目录；目标核验失败不删除本地证据。

## 验证命令与预期信号

### 当前快照基线

    python -B - <<'PY'
    import ast
    from pathlib import Path
    for name in ["scripts/ctc_prealign.py", "scripts/run_pipeline.py", "scripts/streaming_pipeline.py"]:
        ast.parse(Path(name).read_text(encoding="utf-8"), filename=name)
        print("AST OK:", name)
    PY

当前预期：scripts/ctc_prealign.py:1395 非零失败；这是真实阻断，不得标记为通过。

### 实现后静态与回归

    python -B -c 'import ast,pathlib; fs=list(pathlib.Path("scripts").glob("*.py"))+list(pathlib.Path(".").glob("*.py")); [ast.parse(p.read_text(encoding="utf-8"), filename=str(p)) for p in fs]'
    python -B scripts/verify_reference_authority.py
    python -B scripts/verify_tier_discontinuity.py
    python -B scripts/verify_strict_ok.py
    python -B scripts/verify_strict_ctc_ready_import.py
    python -B scripts/verify_reference_only_ctc.py
    python -B scripts/verify_english_mfa_provenance.py
    python -B scripts/verify_prepare_hecheng_english_ctc_ready.py
    python -B scripts/verify_mapping.py
    PYTHONPATH=scripts python -B verify_risks.py
    python -B scripts/verify_pipeline_modes.py
    python -B scripts/verify_ctc_orchestration.py
    python -B scripts/verify_streaming_failures.py
    git diff --check

预期：所有正向命令退出 0；每个注入故障退出非零；失败事务没有父 marker/final manifest；git diff --check 通过。现有 archive 声称的 86/86 结果见 REGRESSION_ARCHIVE.md:7405-7414，只能作为历史记录，必须用当前工作树重新执行。

### 模式与 canary

为 full、ctc_ready、batch_ctc_ready、nvrasr_fallback 各准备最小 fixture，另准备单卡/all-GPU CTC、sharded MFA、strict English 和 direct/staged streaming fixture。每次运行必须保存：冻结 stems、阶段 manifest、CTC/MFA receipt、strict audit、output/filtered 集合和 publish manifest。预期是 input 集合等于成功、filtered、failed 三者互斥并集；任一复制、子进程、审核、上传或目标核验失败都返回非零。

## 注意事项与不变量

- 不得删除、覆盖、reset 或 checkout 当前未提交文件；尤其保留 configs/hecheng_ria_test1.yaml 与三个已修改脚本。
- 不得把 --force 当作成功；它只能继续收集错误。
- 不得以目录存在、单个 TextGrid、可解析旧 manifest 或 v3 marker 证明完成。
- CTC 时间轴必须绑定实际 WAV header；模型 frame 轴不能成为权威 duration。
- reference-only 模式的 required sidecar、.lab、tokens、TextGrid words、punct 和 raw/cn text 必须来自 reference 或确定性变换；ASR 只能是诊断来源。
- 父 marker 只能在所有 shard、manifest、bundle、namespace、时间轴和 publish 检查完成后最后写入。
- 目标目录非空、文件 hash 不符或发布后独立验证失败时必须停止并保留 staging 证据。

## 风险与回滚

Fail-closed 会使旧缓存和历史 partial output 需要重跑，运行成本增加；精确分母会暴露此前隐藏的遗漏；统一 run-specific 路径可能影响依赖扁平 output/ 的脚本；receipt/marker schema 变更会要求显式版本迁移。

回滚只允许按提交级别回滚实现，并保留所有新旧 run-specific artifact、日志、manifest 和 receipt。不得通过删除旧 CTC、aligned、filtered 或 NAS 结果回滚；发布始终写入新版本目录并拒绝非空目标。

## 阻断项

1. 当前工作树 scripts/ctc_prealign.py AST 语法错误。
2. all-GPU preflight 的 allowed namespace 与 .ctc_run_receipt.json 要求矛盾。
3. ordinary mode 仍可复用 partial/stale CTC 或 adjusted 子集并缩小分母。
4. batch/streaming 仍存在失败吞并、错误入队、上传路径不一致或退出码丢失。
5. strict output 与 batch/streaming 目录契约未统一。
6. v3 marker、NVV、缺失 reference 与 [PAUSE] 语义未完成决策。
7. Cases 80–103 缺少当前工作树下完整 CPU/GPU/MFA/NAS canary 闭环。
8. README、config reference、archive 状态与可执行行为不一致。

## 执行清单

- [ ] 记录并保留当前 worktree 改动，修复 ctc_prealign.py 的语法结构。
- [ ] 先加入 R2–R6 的失败回归，再改实现。
- [ ] 完成共同 run contract 与四模式 matrix。
- [ ] 完成 CTC receipt/namespace/marker 原子事务。
- [ ] 完成 partial/stale reuse 与 MFA shard fail-closed。
- [ ] 完成 batch/streaming queue、upload、cleanup、exit propagation。
- [ ] 完成 output receipt、版本化发布和配置 schema。
- [ ] 重新运行现有验证器并修正所有非零结果。
- [ ] 完成四模式和多 GPU/MFA/NAS canary，更新 archive 状态。
- [ ] 复核 git diff --check、工作树、manifest 集合和发布 receipt。

## 就绪判定与门禁结果

判定：**NO-GO**。Sol-high 路由、仓库指令发现、代码/归档证据和规划完整性门禁已通过；但当前语法错误、all-GPU 契约矛盾、分母缩小、batch/streaming fail-open、未决数据策略和缺少真实 canary 使实现与生产发布门禁未通过。本窗口未实现修复，也未启动生产运行或发布。

## 下一窗口启动说明

下一窗口从仓库 /mnt/local_E/MFA_Pause/repo 开始，先读取本文件 handoffs/20260807T091514Z-mainline-all-modes-logic.md，确认当前未提交改动仍在，再执行“当前快照基线”AST 命令。修复必须先补失败回归并保持四种模式的共同分母、receipt、退出码契约，完成所有阻断项和 canary 后才可进入生产发布。

