# MFA 管线批量吞吐优化实施交接

## 元数据

- UTC 时间：2026-09-08T05:30:46Z
- 仓库：`/mnt/local_E/MFA_Pause/repo`
- revision：`38c6d740fc318bf25d2bcdf21bedbe7dda6bbe4d`
- branch：`main`
- 状态：dirty；存在用户的 tracked/untracked 工作，本任务未改动它们
- task slug：`mfa-pipeline-efficiency`
- 交接文件：`handoffs/20260908T053046Z-mfa-pipeline-efficiency.md`
- 规划路由：根代理已确认 `gpt-5.6-sol`、`reasoning_effort: high` 的精确派发

## 目标与任务内容

在保持当前输出质量、语料分母、失败关闭和可复现性约束的前提下，提高批量端到端吞吐。先建立阶段基线，再依次优化批大小与进程预算、减少可证明重复计算、调平 GPU/CPU/I/O 流水；不得以放宽 beam、容忍更多缺失、关闭校验或改变模型来换速度。

## 当前行为与结论

当前管线是混合负载。NVASR/CTC 预对齐显式使用 CUDA；中文 MFA、英文 MFA、能量修正和后处理主要使用 CPU 多进程；音频搬运和发布受 CIFS/NVMe I/O 影响。MFA 因而主要是 CPU 任务，GPU 在本仓库主要服务 CTC/NVASR，并不直接加速 MFA 对齐。

管线已经具备 GPU 生产者与 CPU 消费者重叠，不能把“新增流水线”当首要方案。更应先测量现有流水的空转与队列等待，再处理批次过小、重复模型启动、局部并发池缺少统一预算、失败后重算等问题。

## Facts

1. 默认 CTC 设备是 `cuda:0`，MFA 通过 `num_jobs`、分片和子进程扩展；MFA 默认 `dither=0.0`、`clean=false`、`fine_tune=false`（`scripts/run_pipeline.py:241-266`）。
2. `resolve_num_jobs` 的自动值是逻辑 CPU 数；外部调用没有 stem 提示时不会按小批次封顶（`scripts/run_pipeline.py:640-655`）。
3. MFA 子进程环境把 OMP、MKL、OpenBLAS、NumExpr 线程固定为 1，意图以 MFA 进程级并发防止嵌套超订阅（`scripts/pipeline_utils.py:245-264`）。
4. 大于阈值的中文 MFA 才分片，分片数为 `min(8, cpu_count//4, max(1, stems//200))`；每片 jobs 为总 jobs 除以片数（`scripts/run_pipeline.py:3128-3156`）。13 或 132 stem 批次都走单 MFA 实例而非该分片路径。
5. 中文 MFA 保留 `beam=20`、`retry_beam=80`，并对单个遗漏执行更宽 beam 的有限恢复；该行为属于质量与完整性约束（`scripts/run_pipeline.py:3789-3816`, `scripts/run_pipeline.py:3932-3972`）。
6. 流式资源规划器按 `cpu_budget // cpu_workers` 封顶每 worker 的 MFA 与英文 MFA jobs，但没有向 adjust、英文语料构建和 postprocess 传递同一个子预算（`scripts/streaming_pipeline.py:60-122`）。
7. adjust 在本地盘每批最多启动 32 个进程（`scripts/adjust_ctc_boundaries.py:716-746`）；postprocess 自动值也最多 32（`scripts/postprocess_textgrids.py:19154-19208`）；英文语料构建自动值最多 16（`scripts/align_english_mfa.py:1277-1284`）。
8. 现有 pipelined 模式明确让 GPU workers 执行 NVASR，CPU workers 执行 MFA 与 postprocess，并使用有界队列连接（`scripts/streaming_pipeline.py:4751-4761`, `scripts/streaming_pipeline.py:4901-4928`）。
9. 每个 GPU 批次都会新建 `run_pipeline.py` 子进程并运行到 `normalize_en`，失败最多重试三次（`scripts/streaming_pipeline.py:4374-4518`）；每个 CPU 批次另起子进程从 resample 继续（`scripts/streaming_pipeline.py:4529-4575`, `scripts/streaming_pipeline.py:4694-4713`）。
10. 参考生产配置使用 8 GPU workers、8 CPU workers、13 stem/批、MFA 8 jobs/worker，并关闭 `restore_ctc_cache`（`configs/laria_v5_no_reference_strict_8gpu_20260901_full1000_r24.yaml:14-29`, `configs/laria_v5_no_reference_strict_8gpu_20260901_full1000_r24.yaml:51-72`）。这是配置证据，不代表当前活动进程。
11. 历史 1055-stem、132-stem/批运行启动 8 个 GPU worker 与 8 个 CPU worker（`logs/laria_v5_no_reference_strict_8gpu_20260826_v5.log:1-20`），8 次中文 MFA 报告约 23.6–25.9 秒（`logs/laria_v5_no_reference_strict_8gpu_20260826_v5.log:1184-1279`）。该历史运行只能用于提出瓶颈假设，不能外推当前吞吐。
12. 历史日志显示 GPU 结果陆续进入 CPU 队列后 CPU 并行消费（`logs/laria_v5_no_reference_strict_8gpu_20260826_v5.log:858-880`），证实现有阶段重叠已实际运行。
13. 主机只读快照为 384 logical CPU、192 physical cores、2 NUMA nodes、约 1 TiB RAM、8 张 GPU、4 个可用 NVMe；`/mnt/local_E` 是 CIFS。资源会变化，执行窗口必须重采样。
14. 根代理只读进程快照看到多个 gamedata postprocess worker 长时间各占用约一个 CPU 核；这证明后处理是实质 CPU 消费者，但一次快照不能判断它异常或排名最慢。

## Assumptions

- 用户已确认目标是批量吞吐并保持质量，而非最小单文件延迟。
- 后续实施可以使用固定 1000-stem 代表集做受控基准，并允许创建新的隔离工作区与输出根。
- 当前没有可信的分阶段耗时、队列等待、GPU 利用率和音频时长归一化指标，因此不宣称任何优化比例。

## Decisions

- 优先级为：P0 可观测性；P1 批大小与统一 CPU 预算；P2 内容寻址的阶段复用；P3 基于测量调平流水。
- 保持模型、词典、beam、retry_beam、dither、fine_tune、过滤规则、严格收据和分母不变。
- 以物理核 192 作为首轮 CPU 并发上限；逻辑核仅作为后续可验证候选，不作为默认容量。
- 任何缓存命中必须同时绑定音频、文本、模型树、词典、有效参数、代码与 stem 分区，缺一即重新计算。

## Open Questions

| 问题 | 证据与影响 | owner | 决策路径 |
|---|---|---|---|
| 当前端到端关键路径是 CTC、postprocess、英文 MFA、发布还是队列等待？ | 现有日志缺统一阶段计时；决定 P1 资源投向 | 实施者 | 完成 P0 固定集冷启动基线后按 wall-time 占比排序 |
| 13 stem/批是否因模型和解释器反复启动降低吞吐？ | 每批新起 GPU/CPU 子进程；决定最优 batch size | 实施者 | 对 13/64/128/256 做同源矩阵，比较 stems/h 与 GPU idle |
| 192 物理核或 384 逻辑核哪个是合理预算？ | 主机 SMT=2、NUMA=2；决定 jobs 上限 | 运维与实施者 | 用同一矩阵比较吞吐、上下文切换、iowait，不越过内存门槛 |
| 当前运行应否复用 CTC/MFA 派生产物？ | 参考配置禁用 CTC restore；决定 warm-run 策略 | 数据 owner | 明确 fresh/replay 语义后，仅启用签名完整且验证通过的缓存 |

## 范围、约束与不变量

范围内：阶段指标、统一资源规划、批次矩阵、严格阶段缓存、队列调平、相关配置与测试。范围外：替换声学/CTC 模型、降低 beam、改变对齐或过滤语义、放宽 `allow_partial`/`min_output_ratio`、修改生产数据、直接运行全量生产任务。

必须保留现有 dirty 工作树；实施时基于当前内容新建分支或隔离 worktree。所有新收据写入新命名空间。失败工作区保留、成功发布前 hash/分母验证、MFA anchor capability gate 均不得移除。

## 受影响文件与符号

| 文件/符号 | 计划改动 |
|---|---|
| `scripts/streaming_pipeline.py:60-122` `plan_streaming_resources` | 统一计算 MFA、adjust、英文构建、postprocess 的 per-worker CPU 预算，并输出计划 |
| `scripts/streaming_pipeline.py:4374-4748` GPU/CPU phase | 记录 stage wall time、queue wait、stems、audio seconds、cache hit、retry、publish time |
| `scripts/streaming_pipeline.py:4751-5240` pipelined scheduler | 记录队列水位和 worker idle；按实测调平 batch/worker/queue |
| `scripts/run_pipeline.py:640-655` `resolve_num_jobs` | 接受外层传入的有效子预算，并按 stem 数安全封顶 |
| `scripts/run_pipeline.py:3727-4010` `step_mfa_align` | 暴露 MFA 子阶段指标及缓存签名，不改变对齐参数 |
| `scripts/adjust_ctc_boundaries.py:716-746` | 接受显式 worker cap |
| `scripts/postprocess_textgrids.py:19154-19208` | 接受统一 worker cap 并记录耗时 |
| `scripts/align_english_mfa.py:1277-1284` | 接受统一 corpus worker cap |
| `scripts/pipeline_utils.py:245-264` | 集中资源环境与阶段签名公共逻辑；继续固定 BLAS=1 |
| `config.yaml:66-140` 与任务配置 | 文档化资源预算、指标路径、缓存模式和经基准选出的配置 |
| `tests/test_streaming_resources.py` 及阶段测试 | 覆盖总预算、签名失效、质量门和 resume 行为 |

## Numbered Requirements

1. 为每批输出机器可读的分阶段性能收据，覆盖 stage/copy/CTC/adjust/MFA/en-MFA/postprocess/publish/queue wait。
2. 所有 CPU 子池共享一个显式预算，最坏同时运行进程数不超过预算，BLAS 每进程保持 1。
3. 用固定 1000-stem 集执行可复现的冷启动参数矩阵，选出吞吐最高且通过质量门的组合。
4. 至少比较当前配置和以下候选：`13/8/8/8`、`64/8/8/8`、`128/8/8/16`、`128/8/6/24`、`256/8/4/32`；字段依次为 batch/GPU workers/CPU workers/MFA jobs。
5. 阶段复用只允许在完整签名与输出收据验证成功时命中；失败、过期或跨分区缓存必须失效。
6. 保持同一 stem 分母、接受/过滤分区、tier 标签顺序及边界结果；不得增加 retry、timeout 或缺失。
7. P1 仅在固定集冷启动吞吐相对当前配置至少提升 10%，且三次运行中位数满足质量门时进入生产配置。
8. 所有优化可通过配置回退到当前资源计划与无缓存路径。

## 实施顺序与 ownership

1. 实施者先在 `streaming_pipeline.py` 与 `run_pipeline.py` 增加只读指标收据；依赖：无。
2. 实施者补齐资源规划器和各 CPU 子程序显式 worker cap；依赖：步骤 1 可观察总进程数。
3. 性能 owner 在隔离工作区运行冷启动矩阵；每组合三次，固定 stem 清单、输入 hash、代码 revision、模型和词典。
4. 数据质量 owner 对每次输出执行严格审计与语义比较；未通过的候选直接淘汰。
5. 实施者加入严格的 CTC-adjust、MFA 派生缓存签名和 warm-resume 测试；依赖：步骤 1 的收据结构。
6. 性能 owner根据 queue wait、GPU idle、CPU util、iowait 调整 queue size 与 worker 比例；只有证据支持时才改变 8-GPU 数量。
7. 配置 owner 将胜出值写入新的任务配置，保留旧配置用于即时回滚。

## Requirement-to-Acceptance Traceability

| Req | Observable acceptance | Verification |
|---|---|---|
| R1 | 每批收据具备全部阶段、开始/结束/elapsed、stems、audio_s、queue_wait_s、retry、cache 字段 | schema 单测；一批 smoke 后校验 JSON 完整性与总时长误差小于 2% |
| R2 | planner 报告各池预算；测试枚举 worker 组合均满足并发上限；环境显示四个 BLAS 变量为 1 | `pytest -q tests/test_streaming_resources.py tests/test_run_pipeline_mfa_root.py` |
| R3,R4 | 五组配置各有三份同源冷启动收据与汇总表，包含 stems/h、audio-hours/h、阶段 p50/p95、CPU/GPU/I/O | 基准汇总器拒绝缺 run 或输入签名不同的数据 |
| R5 | 任一音频、文本、模型、词典、参数、代码、stem 分区改变都会 cache miss；原样 replay 才 hit | 参数化缓存单测及一组 cold/warm smoke |
| R6 | 与 baseline 的 eligible/output/filtered/missing 集合完全相同；tier 标签/顺序和边界语义比较无差异；retry/timeout 不增 | 严格审计、`compare_alignment_runs.py`、收据集合 diff |
| R7 | 胜出候选三次 wall time 中位数降低至少 10%，且 R6 通过 | 固定集汇总报告自动判定 pass/fail |
| R8 | 关闭新缓存并选 baseline profile 后，命令与输出语义回到当前路径 | 回退单测与 baseline smoke |

## Verification commands and expected signals

```bash
cd /mnt/local_E/MFA_Pause/repo
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_streaming_resources.py tests/test_mfa_retry.py tests/test_run_pipeline_mfa_root.py tests/test_compare_alignment_runs.py
```

预期：全部通过；没有超预算组合、retry 状态机或 MFA root 隔离回归。

```bash
cd /mnt/local_E/MFA_Pause/repo
python scripts/compare_alignment_runs.py --help
```

预期：比较工具可用；实际基准命令由实现后的 profile runner 输出并连同完整输入签名保存，禁止手工混用不同批次清单。

固定集基准必须记录 `nproc`、`lscpu`、`free -h`、各 NVMe/CIFS 文件系统、`nvidia-smi`，并采样 CPU util/iowait/context switches、GPU util/memory、磁盘吞吐。若 CPU 可用量、GPU 数或挂载与基线不同，本轮不进入横向比较。

## 风险、回滚与 cautions

- 批次增大可能提高显存、RAM、失败重算范围；用 GPU peak memory 小于设备容量 85%、系统 available RAM 大于 20%、无 OOM 作为门槛。
- worker 增多可能跨 NUMA 或放大 I/O 争用；吞吐下降或 iowait/context switches 明显上升即回退上一档。
- 共享缓存最大的风险是把旧轴或旧模型结果当新证据；任何签名不完整都按 miss 处理，不提供宽松兼容。
- 历史 MFA 约 24–26 秒不等于当前关键路径；先测量 postprocess、English MFA、上传和等待，不单独追求 MFA microbenchmark。
- 回滚方式：切回 baseline profile，禁用阶段缓存，保留新性能收据；不删除历史或失败工作区。

## 执行检查清单

- [ ] 从 dirty 主工作树创建隔离实现环境并保留现有修改
- [ ] 实现 R1 指标及 schema 测试
- [ ] 实现 R2 统一资源预算及并发测试
- [ ] 冻结 1000-stem 清单与所有输入/模型/词典/代码 hash
- [ ] 完成五组 cold 矩阵，每组三次
- [ ] 对所有候选执行 R6 质量门
- [ ] 仅对胜出候选实现和测量 strict warm cache
- [ ] 写入新的生产 profile，保留 baseline 回退 profile
- [ ] 复跑目标测试与一批隔离 smoke
- [ ] 审核性能收据、质量收据、资源峰值和回滚结果

## Readiness decision and gate results

规划 READY，实施尚未开始。精确 Sol-high 路由已确认；仓库/分支/revision/status 已记录；AGENTS.md 未发现，已读取仓库 `CLAUDE.md`；事实、假设、决策、开放问题已分离；每项 requirement 均映射到可观察验收与验证；候选路径无碰撞；freshness 复核时 revision 未变，所有引用核心文件 hash 与取证快照一致；只计划并最终创建这一份 handoff，未编辑下游文件。生产 rollout 被 R1–R7 基准与质量门明确阻挡，这属于预期验证门而非规划 blocker。

## Next-window startup instructions

在 `/mnt/local_E/MFA_Pause/repo` 开始下一窗口，先阅读绝对路径 `/mnt/local_E/MFA_Pause/repo/handoffs/20260908T053046Z-mfa-pipeline-efficiency.md`，随后重新记录 git status、硬件与挂载快照。第一项改动只实现 R1 性能收据并运行对应单测；得到固定 1000-stem 冷启动基线后，再进入资源与批次调优。
