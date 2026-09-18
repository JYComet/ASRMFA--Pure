# 八卡资源调度与管线优化实施交接

## 元数据

- 生成时间（UTC）：2026-08-07T08:31:39Z
- 仓库：`/mnt/local_E/MFA_Pause/repo`
- revision：`230e9dbc4a7cbc4f4d7c7814fa1e55e51d1ed40c`
- 分支：`main`
- 写入前工作树状态：7 个既有未提交文件；最终复核时间 2026-08-07T08:38:39Z 时另见 `REGRESSION_ARCHIVE.md` 修改，当前共 8 个未提交文件。该文件未由本交接创建或覆盖，按现有工作保护处理。
- 任务 slug：`optimize-eight-gpu-pipeline-scheduling`
- 交接文件：`handoffs/20260807T083139Z-optimize-eight-gpu-pipeline-scheduling.md`
- 适用指令：未发现 `AGENTS.md`；已读取 `CLAUDE.md`、`README.md`、`config.yaml`
- 规划路由：根代理已派发并验证 `gpt-5.6-sol`、`reasoning_effort=high` 的只读规划工作者

## 1. 目标与具体任务

将当前多入口、批次级并发和 GPU/CPU 流水骨架，规划为一个可恢复、可审计、资源有硬上限的八卡生产调度方案。重点不是简单增加进程数，而是使八张卡持续获得可用的 CTC 工作，同时让 CPU、内存、NVMe 和 CIFS/NAS 不因过量并发成为新的瓶颈。

实施窗口需要完成以下工作：

1. 修复已确认的正确性阻塞，再做性能优化。
2. 建立父调度器到子进程的显式资源契约，确保运行时预算真的生效。
3. 对原始音频使用每卡一个 GPU worker；对 `ctc_ready` 输入跳过 GPU，避免无效占卡。
4. 将 GPU、CPU、上传阶段改为有界队列，并以 NVMe 水位实现背压。
5. 通过 batch/attempt 隔离、阶段 checkpoint、目标端验证和 strict reducer 保证结果不被覆盖或误报成功。
6. 以无 GPU 故障测试、单卡、双卡、八卡和 strict canary 逐级放行。

## 2. 背景与当前行为

### 2.1 实际数据流

代码当前的真实模式多于 README 所描述的八步：

```text
full:
trim → resample → prealign → normalize_punct → normalize
→ normalize_ria → normalize_en → adjust → align → align_en
→ postprocess → strict_ok

ctc_ready:
link → pad_silence → normalize_* → resample → adjust
→ align → align_en → postprocess → strict_ok

nvrasr_fallback:
prealign → pad_silence → normalize_* → resample → adjust
→ align → align_en → postprocess → strict_ok
```

真实阶段定义和三种顺序位于 `scripts/run_pipeline.py:3314-3335` 的 `STEPS` 与 step order；README 的简化步骤位于 `README.md:160-173`，已落后于代码。

GPU 主要用于 NVASR CTC 推理，重采样、文本规范化、边界调整、MFA、英文 MFA、后处理和 strict 审计主要使用 CPU。CTC 的 device 从 CLI/config 进入 `step_prealign`，见 `scripts/run_pipeline.py:803-850`；NVASR 根据显存选择内部 batch，24 GB 级卡取 32、40 GB 以上取 64，并将 `batch_size_s` 限制在 300，见 `scripts/ctc_prealign.py:1565-1585`。

### 2.2 当前并发实现

- `scripts/streaming_pipeline.py:2433-2442` 已提供 GPU/CPU 双队列流水骨架。
- `scripts/streaming_pipeline.py:2559-2563` 创建无上限 `gpu_queue`、`cpu_queue` 并把全部 batch 预先放入队列；`--prefetch-buffer` 和 `--upload-buffer` 没有约束这两个队列。
- `scripts/streaming_pipeline.py:2460-2465` 将 GPU worker 数和 CPU worker 数作为两个独立参数读取。
- 中文 MFA 的 BLAS 线程被固定为 1，避免每个 Kaldi worker 再产生线程乘法，见 `scripts/pipeline_utils.py:135-159`。
- 后处理在未显式给定 worker 数时最多启动 32 个进程，见 `scripts/postprocess_textgrids.py:7214-7221`；八个子进程各自触发时会形成不受全局预算控制的并发。

### 2.3 已确认的正确性问题

1. 父进程在 `scripts/streaming_pipeline.py:1761-1803` 计算 `_effective_mfa_jobs`，但子进程命令只在 `scripts/streaming_pipeline.py:252-278` 重新读取原 YAML，父进程内存中的覆盖值没有形成运行契约。
2. `configs/batch_all.yaml:41-43` 设为每个实例 64 个 MFA jobs；如果八个子进程并发，理论峰值为 512 个 jobs，超过已盘点的 384 个逻辑 CPU。
3. `configs/shayi_huali_batch.yaml:113-118` 仍配置 48 个 CPU workers，`configs/shayi_huali_batch.yaml:135-143` 配置每实例 8 个 MFA jobs；这会与英文 MFA、后处理和 I/O 叠加。
4. `_upload_one_batch` 在 rsync 非零时只打印 warning，未必将 `upload_ok` 置为 false，见 `scripts/streaming_pipeline.py:297-348`。
5. staged 路径在处理失败时清理失败现场，见 `scripts/streaming_pipeline.py:285-291`；上传后无论成功与否也清理本地目录，见 `scripts/streaming_pipeline.py:1640-1651` 和 `1679-1688`。
6. staged 路径在上传完成前就把数据集加入 completed checkpoint，见 `scripts/streaming_pipeline.py:1603-1611`；现有 `_load_checkpoint` 只恢复 dataset 集合，见 `scripts/streaming_pipeline.py:1373-1396`。
7. 各 batch 直接写共享 `output`、`filtered` 和 `ctc_pretg_adj`，见 `scripts/streaming_pipeline.py:304-315`；报告、tone mapping 和 strict manifest 存在覆盖或来源混合风险。
8. streaming uploader 查找普通 `output`/`filtered` 路径，但 strict 输出写到 `workspace/strict_ok_runs/run-id`，见 `scripts/run_pipeline.py:3718-3760`；两者目录契约不一致。
9. all-GPU CTC 的 shard namespace 与 receipt 校验不一致，并且 merge 阶段将 shard 文件直接 move 到父 output，见 `scripts/ctc_prealign.py:1278-1385`。
10. `_run_mfa_sharded` 的 `Popen` 异常路径使用 `_failed`、`_return_codes` 的风险位于 `scripts/run_pipeline.py:1445-1505`。
11. `configs/hecheng_english_mfa.yaml:1-5` 明确禁止 `--force`、`--overwrite`，但 `scripts/streaming_pipeline.py:252-262` 和 `scripts/launch_8gpu.py:91-104` 固定加入这两个参数。
12. 当前暂停记录明确指出未完成最终无 GPU 回归、fresh receipt/evidence 和生产执行，见 `EXECUTION_STATUS_20260807.md:25-37`。

## 3. 宿主资源与运行约束

以下硬件信息来自本次只读盘点及仓库记录，生产窗口仍需 Gate 0 重验：

- 双路 AMD EPYC 9654，约 192 个物理核心、384 个逻辑 CPU、2 个 NUMA 节点。
- 内存约 1 TiB，可用约 970 GiB。
- 可作为工作盘的本地 NVMe 为 `/mnt/nvme0`、`/mnt/nvme1`、`/mnt/nvme3`、`/mnt/nvme4`；`configs/batch_all.yaml:20-25` 的候选注释仍包含容量明显不同的 `/mnt/nvme2`，不能将其纳入 7 TB 工作盘映射。
- `/mnt/Raw` 为 CIFS/NAS；历史 batch 日志记录过 rsync timeout，例如 `logs/batch_all_20260710_152038.log:6209`、`:8037`、`:36224`。
- 历史记录曾使用 8× RTX 4090 D，见 `logs/STATUS_REPORT.md:3-8`；当前沙箱无法连接 NVIDIA 驱动，因此卡型、显存、拓扑和实时占用都不能视为生产已验证事实。
- `configs/shayi_huali_batch.yaml:53-59` 给出历史约 9 files/s/GPU 的期望，不能当作已验收 SLA。

## 4. Facts、Assumptions、Decisions、Open Questions

### Facts

1. `scripts/ctc_prealign.py:1150-1245` 已有 `--all-gpus`、按 stem 分片和 `CUDA_VISIBLE_DEVICES` 映射能力。
2. `scripts/ctc_prealign.py:1565-1585` 已有按显存自动选择 CTC batch 的机制。
3. `scripts/run_pipeline.py:1296-1340` 的 MFA 分片公式会依据 CPU 和 stem 数量限制分片数量。
4. `scripts/pipeline_utils.py:148-159` 将 BLAS/MKL/OpenBLAS/NumExpr 线程固定为 1。
5. `scripts/pipeline_utils.py:601-713` 的版本化发布函数会验证 manifest、文件集合、大小和 SHA-256，并拒绝已有目标。
6. `CLAUDE.md:3-27` 要求由逻辑冲突修复触发的异常追加到 `REGRESSION_ARCHIVE.md`，并保留现有记录。
7. 写入前工作树有 7 个预存未提交文件；最终复核另见 `REGRESSION_ARCHIVE.md` 修改。两组改动都不得 reset、checkout、删除或覆盖。

### Assumptions

1. 生产宿主最终可提供 8 张独立 CUDA GPU，但每张卡的显存和拓扑以 Gate 0 实测为准。
2. 初始 batch 采用 500 stems 作为调度起点，后续按 stem 数和总音频时长双约束调节。
3. 24 GB 级 GPU 初始 CTC 内部 batch 为 32，OOM 时降至 16，再调低 `batch_size_s`；40 GB 级卡可由现有检测逻辑选择更高档位。
4. strict 英文任务保持 fresh run root、reference-only、禁止覆盖和集合守恒不变量。
5. MFA job 的 0.5 至 1.5 GB/进程是当前规划的保守估算，实施时需用 canary 峰值校正。

### Decisions

1. 生产入口收敛为一个 Python 调度器；`launch_multi_gpu.sh` 只保留薄包装，`launch_8gpu.py` 降为兼容入口或明确拒绝 strict 生产执行。
2. 原始音频模式使用 8 个常驻 GPU worker，每卡一个 NVASR 模型实例；`ctc_ready` 模式使用 0 个 GPU worker。
3. 初始 CPU 预算采用 8 个 CPU batch worker，每批 24 个中文 MFA jobs，中文 MFA 全局上限 192；英文 MFA 每批 4 jobs，后处理每批 8 workers。
4. 初始 I/O 预算为 4 个 stage worker、2 个 NAS upload worker；GPU-ready queue 为 8，CPU-ready queue 为 16，upload queue 为 4。
5. 初始 NVMe 映射为 GPU 0/4→`/mnt/nvme0`、GPU 1/5→`/mnt/nvme1`、GPU 2/6→`/mnt/nvme3`、GPU 3/7→`/mnt/nvme4`；最终根据 `nvidia-smi topo -m`、`lspci -tv` 和 `numactl -H` 调整。
6. 70%/60% 作为 NVMe 高低水位：达到高水位暂停 stage，降至低水位再恢复。
7. 每个 batch 使用不可变的 run/dataset/batch/attempt 目录；dataset reducer 完成集合、报告和 manifest 审计后才允许发布。
8. strict 单数据集在 reducer 完成前采用两阶段策略：八卡只负责完整 CTC 事务，随后运行完整 CTC-ready CPU strict 管线，不以不安全的批次重叠换取吞吐。

### Open Questions

| 问题 | 证据、影响 | Owner 与决策路径 |
|---|---|---|
| 八卡型号、显存、PCIe/NUMA 拓扑 | 当前 `nvidia-smi` 不可用；直接影响 CTC batch 和 CPU/NVMe affinity | runtime owner 在 Gate 0 执行硬件盘点；失败则禁止 canary |
| 首个优化对象是 strict 53,998 English 还是多数据集批处理 | strict 的 run-global evidence 与普通 batch reducer 需要不同发布路径 | 业务 owner 冻结首个目标；默认先做 strict canary |
| 是否允许 dataset-level strict reducer | 现有 strict manifest 是 run-global，直接合并 batch 可能破坏 evidence | pipeline/release owner 定义 schema，root 审批后实施 |
| 吞吐、GPU 利用率和完成时限目标 | 只有历史约 9 files/s/GPU，没有正式 SLA | 运行 owner 通过单卡/八卡 canary 建基线；建议以 GPU p50≥85% 作为观察目标，同时记录背压原因 |
| 两个无权威 TXT 的英文 stem | `EXECUTION_STATUS_20260807.md:15-23` 记录 WAV 54,000、TXT 53,998 | data owner 补齐或显式排除；strict 默认保持 expected_count=53,998 |

## 5. 八卡调度方案

### 5.1 资源档位

| 资源 | 初始值 | 约束与调节 |
|---|---:|---|
| GPU worker | 8 | 一卡一个模型；启动前校验 0 至 7 卡、显存和拓扑 |
| CTC orchestration batch | 500 stems | canary 后在 250 至 1,000 范围按时延和显存调整 |
| CTC 内部 batch | 24 GB 卡为 32 | OOM 后按 32→16 降档，并记录显存峰值 |
| CPU batch worker | 8 | 保留 CPU 给英文 MFA、后处理、I/O 和系统 |
| 中文 MFA jobs | 每 batch 24 | 8 batch worker 全局最多 192 |
| MFA shards | 500 stems 为 2 | 每 shard 约 12 jobs，需由实际分片函数确认 |
| English MFA jobs | 每 batch 4 | 全局最多 32 |
| Postprocess workers | 每 batch 8 | 全局最多 64，不能使用未受调度器控制的默认 32×N |
| Stage workers | 4 | 初始按 4 块可用 NVMe 分配 |
| Upload workers | 2 | CIFS 历史 timeout，先保守并支持重试 |
| GPU-ready queue | 8 | 约一轮 GPU 工作量 |
| CPU-ready queue | 16 | 最多积压两轮 GPU 产出 |
| Upload queue | 4 | 防止 NAS 慢导致 NVMe 无限增长 |
| NVMe 水位 | 70%/60% | 高水位暂停 stage，低水位恢复 |

### 5.2 状态机与目录

每个 batch 只允许单向状态转移：

```text
discovered → staged → gpu_running → gpu_verified → cpu_running
→ cpu_verified → upload_pending → uploaded → published
```

异常状态分别为 `failed_stage`、`failed_gpu`、`failed_cpu`、`failed_upload`、`failed_verify`。每个 attempt 必须记录：run ID、dataset、batch ID、attempt、精确 stem set digest、GPU ID、PID、argv、模型/字典 digest、文件数/字节数/SHA-256、阶段时序、return code、timeout/OOM/signal、集合守恒结果。

目录设计要求：

- 输入 staging、CTC 输出、MFA workspace、postprocess 输出和待上传内容属于同一 attempt 根目录。
- 不同 batch 不得直接写共享 `output`、`filtered`、`ctc_pretg_adj` 或跨 batch 报告。
- reducer 只读取已验证 attempt，生成独立 dataset staging；完成全局集合审计后交给 `publish_output_versioned`。
- 失败和未发布 attempt 必须保留，供重试、取证和回滚使用。

### 5.3 背压与 NUMA

生产调度器应使用有界 `Queue(maxsize=16)`，并把 stage、GPU、CPU、upload 四类资源作为独立令牌池。GPU producer 在 CPU-ready queue 达到 16 时阻塞；stage producer 在任一目标 NVMe 达到 70% 时暂停；upload producer 在 upload queue 达到 4 时暂停继续 stage。恢复条件为 queue 消费或 NVMe 降至 60%。

Gate 0 后按拓扑生成 GPU→NVMe→NUMA 映射，使用 CPU affinity 或 NUMA 绑定减少跨节点读写。`/mnt/nvme2` 不得因为配置注释而自动加入映射。

### 5.4 故障恢复

- GPU OOM：保留当前 attempt，记录显存峰值；使用新 attempt 将 CTC internal batch 减半，最多自动重试两次。
- timeout、非零退出、信号退出、缺文件、集合不全或 hash 不一致：状态为失败，不能写 completed，不能删除现场。
- rsync 非零或 timeout：状态为 `failed_upload`，本地结果保留，指数退避重试三次；目标端重读并校验后才算 uploaded。
- 重启只跳过 `published` 且目标端验证成功的 batch；从 `gpu_verified` 或 `cpu_verified` 恢复时不重复已验证阶段。
- 同一 batch 的重试必须使用新 attempt 目录，不得用 `--overwrite` 修补旧 attempt。
- dataset 只有在全部 batch published、stem union 精确、无重复、报告合并唯一且 dataset-level audit 通过后才进入 completed checkpoint。

### 5.5 可观测性

每个 run 输出结构化 `events.jsonl` 和 `run_status.json`。事件字段至少包含 run/dataset/batch/attempt、阶段、worker、GPU ID、开始/结束时间、状态、return code 和错误类型。指标至少包含：GPU 利用率、显存、温度、功耗、files/s、audio-seconds/s、OOM；CPU/RSS/可用内存/MFA job 数；NVMe 空间、queue depth、读写吞吐；NAS stage/upload 吞吐、重试和 hash 时间；各阶段 p50/p95、expected/produced/passed/filtered/error。

## 6. 编号需求

1. R1：收敛为唯一生产调度入口，避免重复启动八份完整主管线。
2. R2：将 MFA、英文 MFA、postprocess、timeout、strict staging 和 GPU 映射作为 child 的显式运行契约，并记录 resolved config hash。
3. R3：原始音频模式启动八个一对一物理 GPU worker，启动前验证 GPU 数量、显存和拓扑。
4. R4：GPU→CPU→upload 使用有界队列和 NVMe 高低水位背压。
5. R5：batch 使用独立 run/dataset/batch/attempt 目录，禁止共享报告和结果直接覆盖。
6. R6：checkpoint 细化到 batch 和阶段；completed 只在发布后目标端验证成功时写入。
7. R7：上传失败、timeout、rsync 非零、文件缺失和 hash 漂移必须失败并保留本地现场。
8. R8：strict 保持 fresh target、reference authority、expected/passed/filtered/report 集合守恒和版本化发布。
9. R9：CPU、RAM、GPU 显存和 I/O 并发均有硬上限；初始中文 MFA 峰值不得超过 192 jobs。
10. R10：所有阶段输出结构化日志、心跳、吞吐、队列、资源和失败原因指标。
11. R11：`ctc_ready` 输入不启动 GPU；只有需要 NVASR CTC 的原始音频进入八卡阶段。
12. R12：完成 receipt namespace、MFA Popen 异常、strict output 路径、上传误报和失败清理等 P0 修复，并通过无 GPU fault tests 后才允许性能 canary。
13. R13：逻辑冲突修复按 `CLAUDE.md:3-27` 追加回归案例，不删除既有 Cases 76、78、83、101 等记录。
14. R14：全过程保留写入前 7 个预存未提交文件，以及最终复核发现的 `REGRESSION_ARCHIVE.md` 改动。

## 7. 有序实施计划、依赖与责任

1. **Gate 0：现场和硬件冻结**（责任：runtime owner；依赖：无）
   - 记录 revision、dirty diff、8 卡型号/显存/拓扑、NUMA、NVMe、RAM、活动进程。
   - 冻结首个数据集、expected stem set、run root 和发布目标。
   - 若 GPU 或拓扑盘点失败，保持 NO-GO。

2. **P0 正确性修复**（责任：pipeline/CTC owner；依赖：Gate 0）
   - 统一 all-GPU receipt namespace，改为 parent staging 后原子切换。
   - 在 MFA shard 启动前初始化异常路径所需的状态。
   - rsync 非零、timeout、缺文件和 hash mismatch 统一转为失败；失败目录保留。
   - 完成上传验证后再写 completed；移除 child 固定的 `--force --overwrite`。

3. **显式资源契约**（责任：pipeline owner；依赖：P0 修复）
   - 增加 resolved run-local config/receipt，显式传递 MFA、English MFA、postprocess、timeout、GPU ID、CPU affinity 和 staging 目录。
   - 记录 config hash 和实际 argv；不得修改用户 YAML。
   - child 启动日志必须显示 resolved 值，不能只显示父进程估算值。

4. **有界调度器与阶段 checkpoint**（责任：scheduler owner；依赖：显式资源契约）
   - 将无界队列改为有界队列，增加 stage/GPU/CPU/upload worker pool 和 NVMe 水位控制。
   - 实现单向状态机、heartbeat、batch-level 原子 checkpoint 和崩溃恢复。

5. **隔离输出与 dataset reducer**（责任：pipeline/release owner；依赖：P0 修复、阶段 checkpoint）
   - 每 batch 生成不可变 attempt 输出。
   - reducer 校验 stem union/intersection、文件集合、report 单 stem 唯一、tone mapping 一致性。
   - strict 任务在 dataset-level audit 后调用版本化发布，不改变 reference authority。

6. **资源与 NUMA 调优**（责任：runtime owner；依赖：有界调度器）
   - 初始采用 8 GPU、8 CPU batch worker、24 中文 MFA jobs、4 English MFA jobs、8 postprocess workers。
   - 增加 GPU OOM 降档、CPU token budget、affinity 和四盘映射。

7. **可观测性**（责任：scheduler owner；依赖：阶段状态机）
   - 输出 JSONL events、run summary、worker heartbeat、queue depth、GPU/CPU/RAM/NVMe/NAS 指标。
   - 事件必须可由 run/dataset/batch/attempt/GPU ID 过滤。

8. **无 GPU 故障测试**（责任：verification owner；依赖：P0 修复和 reducer）
   - 覆盖 receipt 多余/缺失、Popen OSError、rsync 非零、上传中断、checkpoint crash、报告覆盖、重复 stem、队列背压、strict 路径解析。

9. **渐进 canary**（责任：runtime owner；依赖：所有前置步骤和质量闸门）
   - CPU-only dry plan；1 GPU×50 stems；2 GPU×500 stems；8 GPU×每卡500 stems；代表性 strict canary。
   - 仅当所有硬门通过并由业务 owner 批准后进入 full run。

## 8. 要求—验收追踪

| 要求 | 客观验收标准 | 验证方法与预期信号 |
|---|---|---|
| R1、R11 | 原始音频 plan 显示 8 GPU workers；`ctc_ready` plan 显示 0 GPU workers | 运行 `--plan-json`；计划 JSON 中 worker 数和模式匹配 |
| R2、R9 | child receipt 显示中文 MFA=24、English MFA=4、postprocess=8；中文 MFA 总数≤192 | argv/resolved-config fault test；运行时进程计数不超过配额 |
| R3 | worker 的物理 GPU 映射恰为 0 至 7，子进程内部均使用 `cuda:0` | 8 卡 canary receipt 和 `CUDA_VISIBLE_DEVICES` 审计 |
| R4 | queue 上限为 GPU 8、CPU 16、upload 4；消费者暂停时 producer 阻塞且 NVMe 不越过高水位 | bounded queue unit test；queue depth 和磁盘水位事件可见 |
| R5、R8 | batch 目录和元数据互不覆盖；expected=passed∪filtered，交集为空，report 每 stem 一行 | reducer fixture、strict verifier、目标端 manifest 校验均通过 |
| R6 | 上传前 kill 后重启不重跑 `gpu_verified`；未发布 batch 不进入 completed | checkpoint crash-recovery test；恢复日志从已验证阶段继续 |
| R7 | rsync rc=23、timeout、缺文件、hash mismatch 返回非零并保留 attempt | mocked transfer suite；状态为失败且本地目录仍存在 |
| R10 | events 包含阶段时延、队列深度、资源采样、心跳和失败原因 | event schema verifier；每个 worker 有 heartbeat，过期可报警 |
| R12 | 四类 P0 反例和失败清理反例全部阻断成功路径 | 专项 regression suite 全部通过，生产 gate 仍可拒绝未修复路径 |
| R13 | 每个逻辑冲突修复在 `REGRESSION_ARCHIVE.md` 追加现象、根因、符号和复现验证 | 文档审计发现新增案例且旧案例仍存在 |
| R14 | 写入前 7 个预存文件和最终复核发现的 `REGRESSION_ARCHIVE.md` diff 完整保留，新增差异只属于本任务范围 | scoped `git diff` 对比；无 reset/checkout 痕迹 |

## 9. 可观察验收标准

八卡 canary 必须同时满足：

- 8/8 GPU worker 启动且物理映射互异；`ctc_ready` canary 不启动 GPU。
- 无 OOM、无过期 heartbeat、无未解释的 worker signal/timeout。
- GPU 利用率 p50 达到 85%，或事件中给出输入不足、CPU 背压、NAS 背压等可审计原因。
- 中文 MFA jobs 峰值不超过 192；英文 MFA 和后处理也不超过各自全局预算。
- NVMe 使用率不越过 70%，暂停/恢复事件符合 70%/60% 水位规则。
- 任何 rsync warning 不会被当成成功；目标端重读 hash 全部通过。
- expected、produced、passed、filtered、report 集合严格守恒，重复 stem 为失败。
- strict verifier、发布 manifest 和目标端版本化目录验证全部通过。
- 旧 attempt 不被 `--force` 或 `--overwrite` 修补。

## 10. 验证命令与测试用例

以下命令供下一实施窗口执行，本次规划未运行生产任务、GPU canary 或测试套件：

```bash
cd /mnt/local_E/MFA_Pause/repo
git status --short
git diff --check
git diff --stat

nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi topo -m
numactl -H
df -hT /mnt/nvme0 /mnt/nvme1 /mnt/nvme3 /mnt/nvme4 /mnt/Raw

PYTHONDONTWRITEBYTECODE=1 python -c 'import ast,pathlib; [ast.parse(p.read_text(encoding="utf-8")) for p in pathlib.Path("scripts").glob("*.py")]'

python scripts/verify_reference_authority.py
python scripts/verify_reference_only_ctc.py
python scripts/verify_tier_discontinuity.py
python scripts/verify_strict_ok.py
python scripts/verify_strict_ctc_ready_import.py
python scripts/verify_prepare_hecheng_english_ctc_ready.py
python scripts/verify_hecheng_english_ctc_ready_v4.py

python scripts/verify_streaming_scheduler.py --suite unit
python scripts/verify_streaming_scheduler.py --suite fault
python scripts/verify_streaming_scheduler.py --suite reducer

python scripts/streaming_pipeline.py --config configs/shayi_huali_batch.yaml --gpus 8 --cpu-workers 8 --mfa-jobs 24 --plan-json /tmp/mfa-eight-gpu-plan.json
```

故障测试用例及预期信号：

1. receipt 缺少或多出 stem：拒绝 `gpu_verified`，保留 attempt。
2. `Popen` 抛出 `OSError`：返回明确失败，不出现未初始化变量异常。
3. rsync 返回 23 或超时：进入 `failed_upload`，本地目录保留，重试次数可见。
4. 上传后模拟进程崩溃：重启从最后一个已验证阶段恢复，completed 仍为 false。
5. 两个 batch 使用重复 stem 或同名报告：reducer 拒绝发布并报告交集。
6. CPU consumer 停止消费：GPU producer 在队列上限阻塞，NVMe 不继续超过高水位。
7. strict output 位于 `strict_ok_runs`：上传器只能接收 resolved strict staging，不能返回空成功。
8. 一卡 OOM：新 attempt 使用降档 batch，原 attempt 保留并记录显存峰值。

## 11. 约束、不变量、风险与回滚

### 约束与不变量

- 不改变 reference authority、CTC 时间轴、MFA `fine_tune: false`、strict 集合守恒、fresh target、manifest hash 和版本化发布语义。
- 不以关闭 QC、strict audit、hash 验证或强制覆盖换吞吐。
- 不把 `/mnt/nvme2` 当作四块 7 TB 工作盘之一。
- 不把历史 8×4090D 记录当作当前硬件证明；必须在 Gate 0 实测。
- 保留写入前 7 个现有未提交文件及最终复核发现的 `REGRESSION_ARCHIVE.md` 差异，不执行 reset、checkout、删除或覆盖。
- 逻辑冲突修复必须按 `CLAUDE.md:3-27` 追加回归记录。

### 风险

- CTC GPU 加速后 CPU MFA 或 CIFS 上传可能成为主瓶颈，导致 GPU 空闲；必须通过 queue depth、CPU 阶段 p95 和 NAS 吞吐确认原因。
- 过高 MFA 并发可能同时造成内存压力和 Kaldi 进程抖动；先用 192 中文 jobs 上限并以峰值 RSS 校正。
- strict 的全局 evidence 不适合未经 reducer 的批次增量 merge；在 schema 获得批准前保持两阶段路径。
- 多批次输出隔离和 reducer 会增加本地空间与 hash 时间；以 NVMe 高低水位控制，不删除失败现场。

### 回滚

新调度器失败时停止发布，保留 run root、attempt、事件和 checkpoint，回退到只读审计状态。不得把旧 checkpoint、旧 batch output 或固定 NAS 目录直接标记为 completed，也不得用增量覆盖恢复旧目录。入口切换可保留旧入口一个窗口，但旧入口必须拒绝 strict 生产执行或明确提示其不具备新契约。

## 12. 未解决阻塞项

| 阻塞项 | 证据 | 影响 | Owner | 决策路线 |
|---|---|---|---|---|
| 当前真实 GPU 资源未知 | 本次环境中 `nvidia-smi` 无法连接驱动；历史信息仅在 `logs/STATUS_REPORT.md:3-8` | 不能确认显存档位、拓扑和八卡启动条件 | runtime owner | Gate 0 实测；失败保持 NO-GO |
| strict reducer schema 未冻结 | `scripts/pipeline_utils.py:601-713` 假设单次版本化 staging，streaming 路径另有目录约定 | 不安全的批次合并可能破坏全局 evidence | pipeline/release owner | 先定义 manifest schema，再实现 reducer |
| 首个目标数据集未冻结 | `EXECUTION_STATUS_20260807.md:15-23` 只给出英文基线，普通批量配置另有数据源 | 影响 canary 样本、expected set 和资源基线 | 业务 owner | 选择 strict 或普通批量；默认 strict canary |
| 正式吞吐/SLA 未定义 | 仅有 `configs/shayi_huali_batch.yaml:53-59` 的历史估算 | 无法把 GPU p50 和完成时间变成硬放行条件 | 运行 owner | 单卡/八卡 canary 建立 baseline 并批准阈值 |
| 两个缺少权威 TXT 的 stem 未决 | `EXECUTION_STATUS_20260807.md:15-23` | strict expected 集合必须明确为 53,998 或补齐到 54,000 | data owner | 补齐或显式排除后冻结 manifest |

## 13. 执行清单

- [ ] 重新读取本交接文件和 `EXECUTION_STATUS_20260807.md`。
- [ ] 记录 revision、branch、dirty diff 并保护当前 8 个未提交文件，其中 `REGRESSION_ARCHIVE.md` 需单独确认来源。
- [ ] 运行 `nvidia-smi`、`nvidia-smi topo -m`、`numactl -H` 和四块 NVMe 检查。
- [ ] 冻结目标数据集、expected stem set、fresh run root 和发布目标。
- [ ] 完成 P0 修复及 receipt、上传、strict 路径、MFA 异常故障测试。
- [ ] 实现 resolved resource contract、bounded queues、NVMe watermarks 和 batch/attempt state machine。
- [ ] 实现隔离输出、dataset reducer、目标端 hash 验证和阶段 checkpoint。
- [ ] 实现结构化事件、heartbeat、queue depth、GPU/CPU/RAM/NVMe/NAS 指标。
- [ ] 通过 CPU-only、1 GPU、2 GPU、8 GPU 和 strict canary。
- [ ] 审核 `REGRESSION_ARCHIVE.md` 新增案例与旧案例保留情况。
- [ ] 由业务 owner 明确批准 full run 后才启动生产。

## 14. Readiness 决策与闸门结果

- 规划文档完整性：PASS。
- exact Sol-high 路由：PASS；已由根代理显式指定模型和高推理并收到完成结果。
- 仓库与指令发现：PASS；仓库路径、revision、branch、status、`CLAUDE.md`、`README.md` 已记录。
- 证据与引用：PASS；关键入口、函数、配置、资源和风险均有 repository-relative path:line 证据。
- Facts/Assumptions/Decisions/Open Questions 分离：PASS。
- 需求、验收和验证可追踪：PASS；R1 至 R14 均映射到验收标准和测试。
- 新鲜度检查：PASS；写入前 revision/status 与 cited files 快照一致。
- 占位词检查：PASS；无未填充角括号、泛化替换省略或未决项伪装为已完成工作。
- 文件碰撞与写入：PASS；目标文件先检查为空，将通过同目录临时文件再独占改名。
- 现有工作保护：PASS；本轮不修改 7 个预存文件，也不修改最终复核发现的 `REGRESSION_ARCHIVE.md`。
- 生产放行：NO-GO；解除条件为 P0 修复、无 GPU fault tests、真实硬件验证、分级 canary 全通过和 owner 批准。

## 15. 下一窗口启动说明

在 `/mnt/local_E/MFA_Pause/repo` 启动下一窗口，首先打开 `handoffs/20260807T083139Z-optimize-eight-gpu-pipeline-scheduling.md`，再读取 `CLAUDE.md` 和 `EXECUTION_STATUS_20260807.md`。先执行 Gate 0 硬件与工作树盘点，随后严格按第 7 节顺序实施；在第 14 节的生产放行条件满足前，不启动 full run，不把规划内容当作实现结果。
