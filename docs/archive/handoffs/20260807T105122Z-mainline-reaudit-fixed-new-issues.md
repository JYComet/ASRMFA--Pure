# 主线全模式逻辑复审实施交接方案

## 元数据

- 生成时间：2026-08-07T10:51:22Z
- 仓库：/mnt/local_E/MFA_Pause/repo
- 分支：main
- 审查 revision：07a6fa2195ccb3aa9019c9bddaad3eb6afac02ce
- 对照交接：handoffs/20260807T091514Z-mainline-all-modes-logic.md
- 任务 slug：mainline-reaudit-fixed-new-issues
- 规划路由：已先由 gpt-5.6-sol、高推理强度只读规划工作者完成复审，再由主代理核对当前源码证据。
- 工作树约束：审查开始时已有 REGRESSION_ARCHIVE.md、五个脚本改动以及 configs/hecheng_ria_test1.yaml 和上一份交接文件；本次不覆盖、不回退这些改动，只新增本文件。
- 当前结论：**NO-GO**。部分历史问题已修复或收窄，但四种主管线模式、all-GPU CTC、batch/streaming 适配层尚未形成可证明的闭环。

## 目标与具体任务

重新审查上一份方案列出的缺陷，确认当前未提交修复是否真正生效，并识别由这些修复引入或暴露的新问题。后续实现目标仍是让 full、ctc_ready、batch_ctc_ready、nvrasr_fallback 以及 direct/staged/pipelined streaming 都满足同一发布契约：冻结输入 stem 分母；每个 stem 最终只能属于成功、过滤或失败之一；任何复制、子进程、产物完整性、上传、合并或发布失败都必须形成非零顶层结果。

本文件只规划后续实现，不实现代码、不重写测试、不修改配置、不更新回归档案状态。

## 当前工作树与静态结果

当前既有改动为：

- REGRESSION_ARCHIVE.md
- scripts/ctc_prealign.py
- scripts/launch_8gpu.py
- scripts/pipeline_utils.py
- scripts/run_pipeline.py
- scripts/streaming_pipeline.py
- configs/hecheng_ria_test1.yaml
- handoffs/20260807T091514Z-mainline-all-modes-logic.md

对本次审查涉及的五个 Python 文件执行 AST 解析，结果为 AST OK 5。因此上一份方案中的 ctc_prealign.py 缩进/语法阻断已经修复。

git diff --check 仍失败，错误集中在 scripts/streaming_pipeline.py 的新增行，最早约从第 221 行开始，后续涉及第 799、882、943、992、1487、1713、1761、1826、1924、2124、2450、2580、2901 等区域。尾随空白不是业务逻辑缺陷，但当前补丁不能通过基础质量门禁，必须在实现窗口单独清理并复跑。

## 与上一份方案相比已确认修复的问题

以下修复有当前代码证据，但尚不能自动推导出端到端生产可用：

1. scripts/ctc_prealign.py 当前可解析，上一份方案的语法错误已消失。
2. MFA Popen/返回状态变量已在启动前初始化，修复了未启动时引用未绑定变量的问题，见 scripts/run_pipeline.py:1475-1528。
3. MFA 输出集合检查已扩展到缺失、额外及结构非法 TextGrid，见 scripts/run_pipeline.py:1571-1646、1852-1877。
4. --skip-to 现在拒绝不属于当前模式 route 的 step，见 scripts/run_pipeline.py:4002-4010。
5. full 模式 --scan-only 不再直接保留会修改音频的 trim 路径；nvrasr_fallback --scan-only 现在安全返回，见 scripts/run_pipeline.py:4014-4027。后者虽然不产生变更，但也没有产生可供复核的 inventory，见开放问题。
6. --validate 失败现在会加入失败集合并影响退出码，见 scripts/run_pipeline.py:4121-4132。
7. direct streaming 已检查子进程返回码并传播失败，见 scripts/streaming_pipeline.py:975-995。
8. 原有 StreamingPipeline 类中的 prefetch/process/upload 队列门禁已有改进：复制失败不应进入处理队列，处理失败不应进入上传队列，见 scripts/streaming_pipeline.py:780-803、936-967。
9. batch_ctc_ready 主分支现在把缺失音频数据集加入 fail_list，见 scripts/run_pipeline.py:3732-3737。
10. all-GPU shard namespace 检查已经显式允许 .ctc_run_receipt.json，修复了上一份方案指出的“写入但不允许”的字面冲突；但随后的合并循环仍会碰撞，详见阻断项。
11. 单数据集 run_pipeline.py 已尝试使用 run-specific output/filtered 目录并写入 pipeline receipt，见 scripts/run_pipeline.py:3932-3962、4199-4218。
12. staged 上传现在尝试写入 .staging/batch_NNNN，transport 非零返回会被记录为失败，见 scripts/streaming_pipeline.py:331-400。这只完成了隔离和返回值的一部分，尚未完成最终 dataset-level merge 与验收。

因此 REGRESSION_ARCHIVE.md 中 Cases 104、106、109、116、117、118、119，以及 Case 124 的部分路径可以标为“代码层修复存在”；Cases 112、120、121、122、123、125、126 仍只能标为“部分修复/未验收”，不得直接按“已关闭”处理。

## 当前事实与新发现

### P0：会直接阻断执行或造成错误成功

#### 1. Streaming 传入 run_pipeline 的 MFA 参数不存在

scripts/streaming_pipeline.py:275-277、886-889、2580-2583 会向子进程追加 --mfa-jobs 和 --mfa-en-jobs。但 scripts/run_pipeline.py:3520-3578 的 argparse 没有定义这两个选项。当前默认/有效 MFA job 数为正时，子进程会在真正执行前因 unrecognized arguments 退出，导致 streaming 的 ctc_ready/MFA 阶段全部失败。这是对上一份 Case 105 修复的回归性实现错误：参数“向下传递”了，但接收端未提供契约。

#### 2. all-GPU merge 接受 shard receipt 后仍会发生同名文件碰撞

preflight 在 scripts/ctc_prealign.py:1371-1377 跳过 .ctc_run_receipt.json 的目标碰撞检查，但 merge 循环随后会遍历 shard 中的所有文件；scripts/ctc_prealign.py:1392-1407 的跳过集合只覆盖 manifest、summary、marker 和本地字典，没有跳过 .ctc_run_receipt.json。第一个 shard 会把 receipt 移到父目录，第二个 shard 继续移动同名文件时触发目标已存在/碰撞。父 receipt 还会在 scripts/ctc_prealign.py:1495-1508 重新写入，因此 all-GPU 的正常多 shard 路径仍不可接受。

#### 3. all-GPU merge 锁不是原子获取，且合并不是事务性的

scripts/ctc_prealign.py:1380-1385 采用 exists() 后再创建/写入的模式，两个父进程可以同时通过检查。之后 scripts/ctc_prealign.py:1392-1508 逐文件直接写入最终目录，异常时没有回滚到临时 namespace；即使 finally 在 1510-1514 清掉 .merge_lock，父目录仍可能残留部分 bundle。必须改为原子锁、隔离临时合并目录、完整校验后一次性发布。

#### 4. staged 模式没有真正执行文档承诺的最终合并

_upload_one_batch 的注释在 scripts/streaming_pipeline.py:335-339 明确写着“单独 merge step”会把每批 staging 合并到最终输出，但 _execute_staged 在 scripts/streaming_pipeline.py:1761-1770 只逐批上传并清理，没有 dataset-level merge/manifest/receipt 验证。旧的 _merge_to_nas 位于 scripts/streaming_pipeline.py:459-497，但并未被该新 staged 路径用于完成最终发布。因此当前 NAS 上即便每个批次上传成功，也可能只留下隐藏的 .staging/batch_*，没有可消费的最终 output。

#### 5. empty/missing source 会被当成上传成功

_upload_one_batch 在 scripts/streaming_pipeline.py:360-367 对不存在或空的 source 直接 continue，未把“没有任何可上传产物”转成失败；只要其他 source 没报错，upload_ok 仍保持 true，并在 395-400 返回成功。于是 output/filtered/adjusted 全为空的批次可能被 checkpoint 为已发布，违反“非空且 exact-set 验收后才能成功”的契约。

#### 6. streaming 的 run-specific output 与上传器仍不兼容

run_pipeline.py 在 scripts/run_pipeline.py:3939-3962 把输出重定向到 strict run 或 workspace/runs/<run_id>/output。但 _upload_one_batch 只有在配置的 local_output 存在且为空时才调用 _detect_strict_output，见 scripts/streaming_pipeline.py:352-358；若配置目录不存在，检测甚至不会运行。后续 scripts/streaming_pipeline.py:360-365 继续检查 local_dir/output，无法可靠找到实际 run output。该适配层必须读取稳定的结果 receipt，不能靠目录猜测。

#### 7. pipeline receipt 写在发布前，且记录不到最终发布路径

scripts/run_pipeline.py:4199-4218 在 publish 之前写 receipt，scripts/run_pipeline.py:4238-4243 只在内存中更新 _final_output，没有更新或补写 receipt；消费方得到的机器可读证据仍指向 staging，而不是实际 NAS 版本目录。更严重的是 receipt 只记录 failed_steps，没有 failed_stems，也没有检查 input/output/filtered 的互斥并集。

### P1：批处理/恢复/失败聚合仍可能错误报告成功

#### 8. 新配置校验拒绝现有合法 batch 配置

_KNOWN_TOP_KEYS 在 scripts/run_pipeline.py:3396-3411 不包含 batch 和 use_cache。但 configs/batch_all.yaml:11-35 同时使用 use_cache 和 batch，configs/batch_asrmfa_en.yaml:24-30 也使用 batch。校验在 scripts/run_pipeline.py:3619-3629 对 merged config 执行，默认情况下会在 batch 执行前退出。因此新增 R8 校验当前会阻断现有 batch_ctc_ready 配置，而不是仅拒绝错误配置。

#### 9. 配置 schema 仍不完整且 --force 可绕过错误

validate_config 的路径检查在 scripts/run_pipeline.py:3465-3470 是 no-op；校验主要限于顶层，未覆盖 nested key/type；streaming 代码实际读取的 pipeline 也未进入 _KNOWN_TOP_KEYS。此外 scripts/run_pipeline.py:3625-3629 允许 --force 继续执行 schema error。--force 可以用于诊断，但不能消除最终失败状态，也不应让未知/错误 schema 进入执行。

#### 10. staged 流程把 stage copy 失败当作可处理批次

_stage_one_batch 在 scripts/streaming_pipeline.py:187-199 统计 failed_copies，但返回值只有 local_dir、elapsed、missing_audio，调用处 scripts/streaming_pipeline.py:1363-1375 只依据是否抛异常决定 stage_failures。普通复制返回 false 的批次仍会进入 processing queue，后续可能以不完整 CTC bundle 执行。

#### 11. staged 上传清理失败证据，且没有以 process-success 集合作为上传门禁

_execute_staged 在 scripts/streaming_pipeline.py:1761-1770 对每个待上传目录调用 _upload_one_batch 后无条件 _cleanup_one_batch_dir，即使上传失败也删除本地证据。其上传集合基于 staged directory 是否存在，而非经过 output bundle/receipt 验证的 process-success 集合；结合空源 fail-open，会形成失败批次被清理或误发布。

#### 12. 单数据集 staged target 路径重复嵌套

run_single_dataset 在 scripts/streaming_pipeline.py:1353-1355 将 ds_name 设为 nas_output_root.name，而 _upload_one_batch 又在 nas_output_root / ds_name 下建目录，见 scripts/streaming_pipeline.py:345-347。当 nas_output_root 已经是数据集输出目录时，目标会变成 <output>/<output-name>/.staging/...，与调用者预期的 <output>/.staging/... 不一致。

#### 13. batch resume 仍按数量跳过，不按稳定 batch ID 恢复

_save_batch_progress 只持久化 {done, fail, total}，见 scripts/streaming_pipeline.py:1501-1517；恢复时 scripts/streaming_pipeline.py:2124-2165 从重新生成的 CTC batches 内读取 done，按顺序跳过前 N 个，再处理 fallback batches。若输入排序、缺失 stem 或 batch 切分变化，之前成功的批次会被错误映射到新的批次，造成漏处理或重复处理。需持久化 batch_id、stem digest、mode 和 publish state。

#### 14. sequential/already-complete 路径返回 None，main 将其视为失败

run_batch 在 scripts/streaming_pipeline.py:1860-1862 所有数据集已完成时直接 return；parallel<=1 路径在 1982-1984 调用 _run_batch_sequential 后也直接 return。但 main 在 1187-1191 使用 ok = run_batch(args) 并对 falsey 值 sys.exit(1)。因此“无工作”和顺序成功都可能以失败退出，违背已声明的 bool contract。

#### 15. pipelined GPU prefetch 仍 fail-open

scripts/streaming_pipeline.py:2420-2437 对 link_or_copy_file 的异常直接 pass，没有统计或阻止 NVASR 子进程；scripts/streaming_pipeline.py:2704-2712 对缺失音频数据集跳过而不加入 failed_set。GPU 阶段可能在输入不完整时产生 CTC 结果。

#### 16. pipelined 最终成功状态没有汇总 GPU-only failure

GPU worker 在 scripts/streaming_pipeline.py:2822-2831 更新 failed_set，但最终 run_pipelined_batch 在 2896-2908 仅从 CPU worker 的 fail_list 计算 all_ok。如果最后一个失败只发生在 GPU 阶段而没有进入 CPU queue，顶层可能打印全部成功并返回 0。

#### 17. 普通 batch 分支仍缺少统一 receipt/发布契约

batch_ctc_ready 在 scripts/run_pipeline.py:3744-3815 为数据集拼装普通 config、flat workspace/output 并逐 step 执行；它没有复用单数据集 run-specific receipt、冻结 expected stems、严格输出发布或 dataset-level publish manifest。即便 missing audio 已加入 fail_list，partial output/old output/failed upload 仍未被同一 contract 覆盖。

### P2：复用与产物完整性仍不足

#### 18. prealign reuse 仍接受 v3 marker 和不完整 bundle

step_prealign 在 scripts/run_pipeline.py:804-850 已从“任意一个 TextGrid 即复用”改为校验 marker/manifest digest，这是实质改善；但仍接受 legacy v3 marker，并未验证 expected stem exact set、六文件 CTC bundle、receipt、model/dictionary identity 或输入来源身份。一个陈旧但同名的完整局部目录仍可能通过。

#### 19. adjust reuse 只比较 stem，不验证 sidecar/content/provenance

step_adjust_ctc 在 scripts/run_pipeline.py:1291-1338 比较输入 lab stems 与输出 TextGrid stems，解决了单 TextGrid 直接短路；但没有验证 adjusted lab/tokens/punct sidecar、manifest digest、原始 CTC receipt 及内容身份。正确 stem 名称不能证明 adjusted bundle 属于本次输入。

#### 20. link manifest shortcut 只验证 lab 存在

step_link_ctc 在 scripts/run_pipeline.py:2970-3025 的 fast path 主要验证 manifest stem 与 workspace .lab 文件；它没有验证当前 CTC source、完整 sidecar bundle、manifest provenance、音频/文本 exact set。实际链接失败在 scripts/run_pipeline.py:3225-3305 仍有 warning/部分成功语义，需要改为 fail-closed。

#### 21. MFA 普通分母仍由现存 lab 子集推导

虽然 shard output 集合检查已增强，但 ordinary MFA 的 expected stem 仍主要依赖“当前存在的 .lab 文件”；冻结分母比较在 scripts/run_pipeline.py:1768-1795 只有 ctx["expected_stems"] 被填充时才生效。需要把 expected stems 从首阶段贯穿到 MFA、postprocess、filtered 和 publish，而不是让上游缺失文件缩小分母。

#### 22. MFA shard 可保留同 stem 的 stale TextGrid

在 overwrite=false 时，shard 输出如果目标 TextGrid 已存在可能被跳过；末端主要比较 stem 集而非内容与当前 run 身份，见 scripts/run_pipeline.py:1648-1666。因此“同名且集合正确”仍可能是旧内容。

## 历史问题状态矩阵

| 问题/Case | 当前状态 | 结论 |
|---|---|---|
| CTC 语法错误 | 已修复，AST 通过 | 可关闭静态语法阻断，但需继续做 all-GPU canary |
| Case 104 MFA Popen 未初始化 | 已修复 | 需保留故障注入回归 |
| Case 105 MFA jobs 未传播 | 表面修复、实际回归 | adapter 传参，receiver 未定义 CLI，P0 |
| Case 106 force/overwrite 硬编码 | 部分修复 | 默认仍可能为 true；配置 schema 还不接受 pipeline |
| Case 107 all-GPU TOCTOU | 未修复 | 非原子锁、非事务 merge，P0 |
| Case 108 batch 上传碰撞 | 部分修复 | per-batch staging 存在，但无最终 merge，P0 |
| Case 109 upload 前 checkpoint | 部分修复 | 新 staged path 延迟部分 checkpoint，但空上传和其他 adapter 仍可假成功 |
| Case 110 strict output mismatch | 未修复 | output detector 依赖不存在/空的 configured path，P0 |
| Case 111 batch resume | 未修复 | 数量恢复替代稳定 batch ID |
| Case 112 all-GPU receipt namespace | 字面冲突已修复，流程仍失败 | shard receipt merge 同名碰撞，P0 |
| Case 113 prealign stale shortcut | 部分修复 | marker digest 增强，v3/完整 bundle/provenance 仍缺 |
| Case 114 adjusted CTC stale shortcut | 部分修复 | stem equality 增强，sidecar/provenance 仍缺 |
| Case 115 link manifest shortcut | 部分修复 | 只验证 lab，不验证完整 source identity |
| Case 116 cross-mode skip-to | 已修复 | 需保留 CLI matrix 测试 |
| Case 117 full scan-only 改音频 | 已修复 | fallback scan 仍没有 inventory |
| Case 118 validate 失败不影响退出码 | 已修复 | 需确认 force 不能掩盖最终失败 |
| Case 119 direct child return code | 已修复 | 需保留非零子进程测试 |
| Case 120 prefetch copy failure | 原有类已修复，其他路径未修复 | _stage_one_batch、pipelined GPU 仍 fail-open |
| Case 121 failed process 上传 | 原有类已修复，staged path 未闭合 | 要求 process-success/output receipt 门禁 |
| Case 122 batch 返回 None | 未修复 | sequential/already-complete 仍 falsey |
| Case 123 CPU upload failure | transport 返回值已处理 | 空源、清理和最终 merge 仍错误 |
| Case 124 缺失音频 | 主 batch 分支已计 fail | streaming/pipelined 其他入口仍跳过 |
| Case 125 flat output isolation | 单数据集部分修复 | batch/streaming 未消费 run receipt |
| Case 126 config schema | validator 已添加但不可用 | 拒绝合法 batch/use_cache，且 nested/path 不完整 |

## 假设

- 冻结分母中的每个 stem 必须最终属于且只属于 output、filtered 或 failed。
- 空 output、缺失 sidecar、部分复制、子进程非零/超时/信号退出、上传失败、目标校验失败都必须为失败，而不是 warning 或跳过。
- --force 允许继续收集诊断，但不能把 schema、阶段或发布失败转成 0。
- 可复用缓存必须证明 exact stem set、完整 bundle、内容/模型/字典身份和 source provenance；无法证明时 fresh rerun 或明确失败。
- 静态审查不能关闭真实 GPU、MFA 模型、CIFS/NAS 和音质/对齐质量风险，必须用 canary 证据关闭。

## 决策建议

1. 保留全部当前未提交改动，分阶段修复；不使用 reset、checkout 或删除历史输出。
2. 定义一个稳定位置的 run-result receipt，内容包括 run id、mode、route、输入/成功/过滤/失败 stem、实际 output/filtered/audit 路径、最终 publish 路径和集合摘要。
3. 让 main、batch、streaming、launcher 都只消费该 receipt，不再猜测 local_dir/output 或最近修改目录。
4. all-GPU CTC 在隔离临时 namespace 内合并，receipt/manifest/summary/marker 按“验证完成后最后写入”发布；shard receipt 不进入父目录最终 bundle。
5. 删除或显式迁移 legacy v3 marker；旧 marker 不能作为 v4 完成态。
6. 配置 schema 应包含现有合法 batch、use_cache、pipeline 等字段，执行前做 nested 类型/未知键/模式依赖校验；--force 不绕过 schema 错误。
7. staged upload 必须先验证每批非空、exact stems 和 hash，再做 dataset-level deterministic merge；失败时保留本地证据，只有最终发布 receipt 成功才 checkpoint。
8. resume 使用稳定 batch ID + stem digest + mode，不按“前 N 个批次”恢复。

## 未决问题

| 问题 | 影响 | 决策前不得假设 |
|---|---|---|
| legacy v3 marker 是否继续支持 | 继续支持会保留陈旧 bundle 风险 | 不能默认视为 v4 完成态 |
| filtered stem 是否算发布成功 | 决定 denominator union 和最终统计 | 不能把 filtered 直接当 success |
| batch/streaming 是否必须 strict evidence mode | 决定是否强制完整 receipt/发布契约 | 不能混用弱契约和 strict 目录 |
| streaming staged 输出的最终目录层级 | 影响 nas_output_root 与 dataset name 组合 | 不能接受当前重复嵌套路径 |
| 缺失参考文本、[PAUSE] 映射和 NVV 策略 | 影响 stem 分类及音频/CTC authority | 保留现有规范，交由数据负责人决定 |

## 范围与受影响文件

范围包括四种 run_pipeline.py route、CTC 单卡/all-GPU receipt 与合并、prealign/link/adjust/MFA 复用、batch 分支、streaming direct/staged/pipelined 队列和发布、launcher 的退出码，以及回归档案状态核验。

受影响符号：

- scripts/ctc_prealign.py：all-GPU preflight/merge/lock、单卡与父 receipt、marker finalization。
- scripts/pipeline_utils.py：CTC/pipeline/publish receipt writer 和 bundle/marker helpers。
- scripts/run_pipeline.py：validate_config、四种 route、step_prealign、step_adjust_ctc、step_link_ctc、MFA shard/align、batch 分支、最终 receipt/publish。
- scripts/streaming_pipeline.py：_stage_one_batch、_upload_one_batch、_detect_strict_output、_execute_staged、run_batch、_run_gpu_phase、_run_cpu_phase、run_pipelined_batch。
- scripts/launch_8gpu.py：sequential/parallel 子进程退出码和 shard 完成判定。
- REGRESSION_ARCHIVE.md：Cases 104–126 的状态语言，必须等实现与测试通过后再更新。

## 编号实现要求

1. 在首个会改变输入的步骤前冻结 expected stems、source digest 和 route。
2. 全流程强制 input = output ∪ filtered ∪ failed，三者两两互斥；集合不守恒时非零退出。
3. 统一 pipeline result receipt，并将它放在适配层始终可发现的稳定 workspace 位置。
4. 所有 CTC 成功态必须具备完整六文件 bundle、manifest、summary、run receipt 和 marker；marker 必须最后写入。
5. all-GPU 使用原子 lock 和临时合并目录；忽略 shard receipt 的移动，父 receipt 只写一次。
6. prealign/link/adjust/MFA 复用必须验证 exact set、完整 sidecar、内容身份、model/dict/source provenance。
7. 为 MFA job override 提供 run_pipeline 正式 CLI 或临时 validated config，并增加 argparse 级测试。
8. stage copy、GPU prefetch、child process、audit、upload、destination verify 的失败必须阻断下一阶段并进入聚合状态。
9. 空 source 或零 output 不得返回上传成功；失败批次不得 cleanup，除非失败证据已持久化。
10. staged publish 必须完成 deterministic dataset-level merge、最终 manifest/hash verification 和原子 publish 后才能 checkpoint。
11. batch progress 记录稳定 batch ID、stem digest、mode、status 和发布 receipt，恢复只执行未完成批次。
12. 所有 sequential、parallel、pipelined、already-complete、no-work 路径返回明确 bool 并由入口传播。
13. 修正配置 schema 的合法字段、nested 类型、未知键、必填路径和模式依赖；--force 不掩盖最终失败。
14. 运行 CPU fault suite 后，再执行单卡 CTC、all-GPU CTC、sharded MFA、四种模式、direct/staged/pipelined 和 NAS canary，才能更新 archive closure。

## 有序实施计划

1. **冻结当前状态并补阻断回归。** 先保留工作树；增加两个 all-GPU shard receipt、正 job CLI、合法 batch config、空上传、stage copy failure、strict/non-strict output discovery、顺序/无工作返回值的 CPU-only 测试。
2. **修复配置入口。** 补齐 batch、use_cache、pipeline 和实际配置字段，建立 nested schema 与 fatal validation；确保 configs/batch_all.yaml 能到达 batch 分支。
3. **建立稳定 receipt。** 明确 schema、路径和集合语义；receipt 在 workspace 固定位置生成，发布完成后补写最终 publish 路径和验证摘要。
4. **修复 all-GPU CTC 事务。** 统一 shard/parent receipt 命名和 namespace；用原子 lock、临时目录、完整 bundle 校验、receipt-last、失败清理/保留策略替代直接移动。
5. **收紧复用与 MFA 分母。** 将 frozen expected stems 贯穿 prealign/link/adjust/MFA/postprocess；拒绝 v3/partial/stale/same-name 产物，验证内容和 provenance。
6. **修复 streaming child contract。** 在 run_pipeline.py 正式支持 MFA job overrides；所有 child command 通过 argparse smoke test；adapter 从 receipt 解析实际 output。
7. **完成 staged publish。** stage 失败不入 process；process 失败不入 upload；每批 exact/nonempty 验证；dataset-level merge 只合并已验证批次；失败保留本地证据。
8. **修复 resume 和聚合。** 以 batch ID/digest 恢复；修正 sequential/already-complete bool；把 GPU-only、missing-audio、stage/upload failure 统一纳入顶层状态。
9. **修复 launcher 与基础质量门禁。** sequential launcher 汇总非零；清理 streaming_pipeline.py 尾随空白，使 git diff --check 通过。
10. **分级验证和状态更新。** 先跑 CPU-only fault suite，再跑真实单卡/all-GPU/MFA/NAS canary；只有 receipt、集合、退出码和发布目录全部满足标准后，才更新 REGRESSION_ARCHIVE.md Cases 104–126。

## 客观验收标准

- 所有修改脚本 AST 解析通过，git diff --check 通过。
- 合法 batch config（含 batch、use_cache）不被 schema 拒绝；未知/错误 nested key 在执行前非零。
- streaming 正 job 数 child 命令可被 run_pipeline.py --help/argparse 接受。
- 两个以上 mock all-GPU shard 可合并成一个 exact parent bundle，不发生 receipt collision、TOCTOU 或半成品 marker。
- 任意 stale marker、partial bundle、wrong digest、link copy failure、invalid TextGrid 都不能形成成功 shortcut。
- receipt 可证明 input/output/filtered/failed 的互斥并集，并记录真实最终 publish path。
- strict 与 non-strict 两种 output redirect 都由 receipt 正确发现，不依赖最近修改目录。
- 空上传、缺 output、rsync/copy/timeout、最终 merge collision 都返回非零且保留失败证据。
- staged 目录在 dataset-level merge 与 hash/manifest 验证前不会被视为最终发布。
- sequential、parallel、pipelined、already-complete、no-work 的返回值与进程退出码符合成功/失败语义。
- 非连续 batch resume 只运行准确的未完成 batch ID。
- full、ctc_ready、batch_ctc_ready、nvrasr_fallback 及 direct/staged/pipelined streaming 都能生成可审计 receipt；生产放行前仍需 GPU/MFA/NAS canary。

## 验证命令与测试用例

### 当前快照已执行

    python -B -c 'import ast,pathlib; files=[pathlib.Path("scripts/ctc_prealign.py"),pathlib.Path("scripts/run_pipeline.py"),pathlib.Path("scripts/streaming_pipeline.py"),pathlib.Path("scripts/launch_8gpu.py"),pathlib.Path("scripts/pipeline_utils.py")]; [ast.parse(p.read_text(encoding="utf-8"), filename=str(p)) for p in files]; print("AST OK", len(files))'
    git diff --check

当前结果：AST 通过；git diff --check 非零，需清理 streaming 新增尾随空白。

### 实现后必须新增或执行

    python -B -c 'import ast,pathlib; [ast.parse(p.read_text(encoding="utf-8"), filename=str(p)) for p in pathlib.Path("scripts").glob("*.py")]'
    git diff --check
    python -B scripts/verify_reference_authority.py
    python -B scripts/verify_tier_discontinuity.py
    python -B scripts/verify_strict_ok.py
    python -B scripts/verify_strict_ctc_ready_import.py
    python -B scripts/verify_reference_only_ctc.py
    python -B scripts/verify_english_mfa_provenance.py
    python -B scripts/verify_mapping.py
    PYTHONPATH=scripts python -B verify_risks.py

必需的 CPU-only 故障用例：

1. all-GPU 两个 shard 各含 .ctc_run_receipt.json，验证无同名碰撞且父 receipt 只生成一次。
2. 并发父 merge 获取 lock，验证只有一个 owner，失败不留下完成 marker。
3. bundle 验证成功后 receipt 写入失败，验证不会留下可复用 marker。
4. prealign v3 marker、partial v4 bundle、stale manifest、wrong model/dict digest。
5. link 阶段单个 copy 返回 false 或缺 sidecar。
6. adjust TextGrid stem 正确但 sidecar/provenance 过期。
7. MFA overwrite=false 下同 stem stale TextGrid 内容。
8. configs/batch_all.yaml 与 batch_asrmfa_en.yaml 的 schema 通过，未知 batchx 失败。
9. streaming 正 --mfa-jobs/--mfa-en-jobs child argparse smoke test。
10. strict/non-strict run-specific output receipt discovery。
11. empty output、rsync failure、copy failure、merge collision、destination hash failure。
12. stage copy failure、missing audio、GPU prefetch exception。
13. sequential success、already complete、pipelined GPU-only failure 的顶层返回值。
14. 非连续 batch resume，验证按 batch ID/digest 而非前 N 个恢复。

GPU/NAS canary 必须覆盖单卡 CTC、all-GPU CTC、sharded MFA、四种主模式、direct/staged/pipelined streaming、传输失败、版本化发布和最终 receipt 消费。

## 风险与回滚

主要风险是拒绝旧缓存导致重跑、run-specific 目录增加存储、receipt/版本化发布改变下游消费路径，以及严格失败语义暴露此前静默遗漏的 stem。应先在小型 fixture/canary 上验证，再扩大规模。

回滚只能使用版本化提交或保留的历史目录；不得用 git reset --hard、git checkout --、递归删除 workspace/NAS 或覆盖已有发布目录。保留 shard 日志、failed batch 目录、staging、receipt、manifest 和已发布版本，便于审计与恢复。

## Readiness blockers

1. streaming 传入 run_pipeline.py 的 MFA job 参数未定义，正 job 配置会直接失败。
2. all-GPU shard receipt 同名移动碰撞。
3. all-GPU lock 非原子，merge 非事务性。
4. run-specific output 与 streaming 上传器不兼容。
5. pipeline receipt 在 publish 前写入且不记录最终发布路径/failed stems。
6. staged path 没有最终 dataset-level merge，空 source 可假成功。
7. stage copy、pipelined GPU copy、missing-audio 仍有 fail-open 路径。
8. staged 上传失败后无条件 cleanup，失败证据可能丢失。
9. 合法 batch/use_cache 配置被新 schema 拒绝，--force 又可绕过错误。
10. batch resume 仍按数量，sequential/already-complete 返回 None。
11. batch 分支缺少统一 run-specific denominator/receipt/publish contract。
12. prealign/link/adjust/MFA 仍存在 partial/stale/same-name 复用风险。
13. git diff --check 当前失败。
14. 尚无本次工作树对应的 GPU/MFA/NAS canary 证据。

在以上阻断项关闭并完成分级验证前，主管线四模式及其 batch/streaming 适配层不得进入生产运行，也不得把 REGRESSION_ARCHIVE.md 中对应案例统一标为已关闭。
