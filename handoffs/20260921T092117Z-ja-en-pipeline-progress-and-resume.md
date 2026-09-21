# 日英 ASR → MFA → TTS 管线：项目概况、停止时进度与续接记录

记录时间：2026-09-21 09:21 UTC。用户要求“先记录当前进度和任务目标，总述项目概况，结束当前实现”。已停止实现代理；本文是状态记录，不代表验收通过，也不授权自动恢复执行。

## 项目概况与任务目标

本项目在既有中文／英语 MFA 工程旁新增独立日英数据管线，从游戏原始音频和台本生成可追溯的 TTS 训练记录。完整目标是：

```text
原始 WAV + 台本 + speaker/uid
  → inventory 与保守音频准备
  → 多家族 ASR 盲听
  → 按 occurrence 选择并冻结 reading
  → 日语文本前端、字符映射、semantic phone/mora graph
  → 唯一 ASCII alias 与单发音词典
  → Qwen 全句 lexical anchors
  → Japanese MFA / English US ARPA 分语言对齐
  → mixed seam 双侧重跑或拒收、严格合并
  → TTS JSONL + TextGrid + 独立证据校验
```

实施依据是原始完整交接方案：
[20260921T050932Z-awesome-japanese-nlp-ja-en-mfa-pipeline.md](/mnt/local_E/MFA_Pause/repo/handoffs/20260921T050932Z-awesome-japanese-nlp-ja-en-mfa-pipeline.md)。

核心边界：

- MFA 是唯一生产 phone timing 后端；Qwen ForcedAligner 只提供 lexical anchors。
- 日语和英语各用独立模型、inventory、词典和运行目录，不制造伪双语 acoustic model。
- baseline ASR 为 Qwen、Whisper、Reazon 三家族；同家族扩展模型不增加票数。`qwen_only_dev` 只能作为开发验证。
- reading 依次使用带来源的人工 override、origin exact surface/kana lexical support、跨家族 kana consensus；共享 G2P 推导必须标记，不能声称声学读音真值。
- 每个 occurrence 的 alias 唯一，词典中恰一条锁定发音；最终 phone 必须追溯到原始 MFA TextGrid 和源音频整数 sample。
- mora 与 phone 是多对多关系，不均分长元音来伪造拍内边界；文本 accent 与测量 F0 分层。
- Julius 是默认关闭的独立诊断分支，不回写 MFA/TTS、不参与 reading vote。
- 缓存必须绑定输入、代码、配置、模型、前端选项和运行环境；每个 UID/unit/run 精确归入 verified、rejected、unresolved。

## 工作区和保存状态

| 项目 | 当前值 |
|---|---|
| 实现工作区 | `/home/user/JapMFA` |
| 分支 | `codex/ja-en-full-pipeline` |
| 基线 commit | `ff2e2f34088e61ba45f08d29d32e6f2dd3c72a44` |
| 原仓库 | `/mnt/local_E/MFA_Pause/repo`，本任务按只读保护 |
| 外部证据与运行产物 | `/home/user/ja-en-task-artifacts` |
| Git 状态 | 未提交；旧文件 3 个修改、新增实现／配置／测试等文件保留在工作区 |
| 提交／发布 | 未 commit、未 push、未创建 PR、未生产发布 |
| 停止状态 | core 与 verifier 实现代理已中断，其余代理已完成；停止时未发现本任务 CLI/ASR/MFA/pytest 子进程仍在运行 |

停止前保存了 50 个变更文件的路径、大小与 SHA-256，以及 tracked diff、branch、HEAD、status：
[停止快照目录](/home/user/ja-en-task-artifacts/stop-snapshot-20260921T092117Z)。此快照先于本文创建，不包含本文。`tracked-diff.patch` 不包含 untracked 新文件；新文件本体仍在工作区，勿清理。

原仓库六个受保护文件在停止时再次逐个校验，全部哈希不变：原 Qwen normalization、reference macro 测试、macro probe 配置、retire 脚本以及两份原始 handoff。检查基准为
[protected-files.sha256](/home/user/ja-en-task-artifacts/protected-files.sha256)。

## 已落地的实现

以下表示代码已存在且有局部验证，**不表示完整正常入口已验收**。

| 模块 | 当前实现与主要文件 |
|---|---|
| 入口、schema、阶段编排 | `scripts/run_ja_en_pipeline.py`、`scripts/ja_en_schema.py`；阶段 registry、receipt、identity、resume、run lock、逐 UID 编排正在最后联调 |
| 供应链与音频 | `verify_ja_supply_chain.py`、`ja_audio.py`；资源 pin/hash/license gate，目录树与运行环境绑定，source/train/alignment 收据，明确 PCM/resample recipe |
| ASR 与 reading | `ja_asr_provider_worker.py`、`ja_asr_crossval.py`；三种 profile、provider 隔离调用、family 去重、候选选择、人工 override、English dictionary pronunciation 锁 |
| 日语前端 | `ja_frontend.py`、`ja_text_layers.py`、`check_ja_frontend.py`；固定 frontend provider、显式 API flags、文本层/span、候选投影和锁定重构 |
| phone/mora 与 aliases | `ja_phone_adapter.py`；semantic graph、模型 native adapter、唯一 alias、唯一 pronunciation、多对多 mora 关系 |
| 跨阶段数据连接 | `ja_en_stage_inputs.py`；四个 API：`prepare_anchor_requests`、`prepare_alignment_requests`、`prepare_merge_requests`、`assemble_tts_rows` |
| anchors、MFA、merge | `ja_en_anchors.py`、`align_japanese_mfa.py`、`merge_ja_en_mfa.py`；纯语言和 mixed 双 pass、分语言 MFA、seam/retry、strict ledger |
| TTS 与独立校验 | `ja_tts_export.py`、`verify_ja_en_tts.py`、`ja_canary_gate.py`；JSONL/TextGrid、三条 sample 轴、外部证据重开检查、release gate；仍有联调未结项 |
| 可选 Julius | `run_julius_diagnostic.py`；固定 converter、独立诊断 namespace、缺资产返回 unavailable、不回写生产产物 |
| 配置和文档 | `configs/japanese_english_tts.yaml`、`requirements-ja-frontend.txt`、`environment-ja-frontend.yml`、`docs/JA_EN_PIPELINE.md`；README 与旧 Qwen 文档增加独立入口说明 |
| 下载器 | `scripts/download_models.py` 新增显式 `--only ja-en`；旧默认 MFA/HF 模型列表经 AST 对照未改变 |

停止前最后一批已完成修正：

1. 真实 `今日` 原 contextual reading 为 `キョウ`，Qwen/Whisper 输出 `こんにち` 时，候选投影可选择 `コンニチ` 的跨家族 consensus。多 token 非 exact 仅在明确上下文锚点之间的唯一单 token gap 投影，歧义保留 unresolved。
2. W2 把句号等标为 `morph_punct`，保留文本 span；W1 reading 集合排除这些标点，避免伪 unresolved reading。
3. English 读音锁字段已两侧对齐：selector 写 `chosen_pronunciation`、`pronunciation`、`native_phones`，frontend 能读取并输出 locked pronunciation。
4. mixed anchors 每个词单元只取其语言 pass。另一个 pass 的 item 可以跨越非目标语言内部词边界；seam 仍要求两 pass 在真实字符边界有 item edge。
5. 上述 mixed 回归已使用真实缓存双 pass 输出，并保存为可移植 fixture `tests/fixtures/frozen_mixed_qwen_worker_output.json`。

## 已得到的真实运行证据

开发样例是合成日语和公开 Qwen 英语 demo，不是游戏语料、独立人工 gold 或生产质量证明。

- 固定 `pyopenjtalk-plus` commit `9e4bf25324ac135dfc81ca64aed2fa6a48b83304` 与固定 Open JTalk/HTS 子模块成功构建，独立环境可运行。已实测 provider 唯一性、caller/morph/NJD span 差异及 18 个读音／adapter 样例。
- 真正加载本地原生 Qwen3-ASR-1.7B 与 Qwen3-ForcedAligner-0.6B，完成日语合成音频 ASR/anchors；英语公开 demo 裁片识别为 `People.`。
- Japanese MFA v3.0.0 与 MFA 3.3.9 真实 alignment 通过；`東京` 得到 native `t oː c oː`。实际 dither=0 和 alias 不分词选项已确认。
- English US ARPA v3.0.0 真实 alignment 通过：公开 demo 36 个 aliases、133 个 phone 区间，无缺失／额外／未知 phone。
- 实际 mixed Qwen 两 pass 已产生输出，修正后的 anchor mapping 可以回放通过，包括 English-pass `きました` 跨越多个日语词单元的情况。
- 固定 Julius converter 的实际源码已与真实 frontend reading 做表示验证；**未运行完整 Julius binary/model alignment**。

主要证据文件：

- [frontend-installed-build-evidence.json](/home/user/ja-en-task-artifacts/frontend-installed-build-evidence.json)
- [frontend-adapter-runtime-cases.json](/home/user/ja-en-task-artifacts/frontend-adapter-runtime-cases.json)
- [qwen-live-smoke.json](/home/user/ja-en-task-artifacts/qwen-live-smoke.json)
- [mfa-asset-evidence.json](/home/user/ja-en-task-artifacts/mfa-asset-evidence.json)
- [Japanese MFA 原始 TextGrid](/home/user/ja-en-task-artifacts/mfa-live-smoke/output-v4/tokyo.TextGrid)
- [English MFA 原始 TextGrid](/home/user/ja-en-task-artifacts/mfa-en-live-smoke/output/demo.TextGrid)
- [root-julius-representation-smoke.json](/home/user/ja-en-task-artifacts/root-julius-representation-smoke.json)
- [mixed 双 pass 原始输出](/home/user/ja-en-task-artifacts/final-cli-proof/isolated-mixed-anchor/stages/anchors/qwen_worker_output.json)

可复用运行环境：frontend `/home/user/ja-en-task-artifacts/frontend-venv/bin/python`，Qwen `/home/user/miniconda3/envs/asr/bin/python`，MFA `/home/user/miniconda3/envs/mfa-dev/bin/python`。原生 Qwen 模型位于 `/mnt/nvme3/models/Qwen3-ASR-1.7B` 和 `/mnt/nvme3/models/Qwen3-ForcedAligner-0.6B`。官方日英 MFA zip/dict 和 metadata 已存外部证据目录。

## 测试状态：必须区分历史结果与最终验收

- root 曾运行一次较早集成版本全量 pytest：**1518 passed**。随后进行了大量修正，此数字不代表停止时版本。
- 后续代理曾报告不同中间版本全量通过；尚未由 root 对停止时完整版本执行最终全量验收。
- root 最近亲自运行 anchors、bridge、MFA runner、seams、native adapter、mora 的定向集合：**29 passed，4 skipped**，跳过的是可选真实环境测试。
- W2 最新 frontend/Julius 定向集合报告 **36 passed**。
- W1 最后标点 participation 修正报告 **22 passed，14 skipped**；此前 English 锁修正也已测试。
- W3 mixed anchor 与 bridge 等集合报告 **26 passed**；可移植 fixture 修正后相关集合 **14 passed**。
- **没有最终全链正常 CLI → TTS → 独立 verifier 成功证据；没有最终 resume/Julius 切换/篡改整链验收。**

旧仓库部分测试依赖 ignored cache；已从原仓库复制四个旧 cache JSON 到工作区，仅供旧测试使用。新日英管线没有消费这些缓存。详情见外部 [progress.md](/home/user/ja-en-task-artifacts/progress.md)。

## 正常入口最后运行状态

配置及输入已准备：

- 双 UID 日语／英语：[config.yaml](/home/user/ja-en-task-artifacts/final-cli-proof/config.yaml)
- 三 UID 加 mixed：[config-mixed.yaml](/home/user/ja-en-task-artifacts/final-cli-proof/config-mixed.yaml)
- 样例来源说明：[fixture-provenance.json](/home/user/ja-en-task-artifacts/final-cli-proof/fixture-provenance.json)
- 最后失败 stderr：[run-core.err](/home/user/ja-en-task-artifacts/final-cli-proof/run-core.err)

已观察到的最后一次正常 CLI：inventory/audio/asr/reading 到达 COMPLETE，frontend 为 PARTIAL，semantic 为 BLOCKED，随后 anchor 输入组装以 `dictionary_roundtrip_failed` 结束。具体是 English 缺 `chosen_pronunciation` 和日语句号 `pau` 被当作 lexical phone。上述两类输入缺陷刚修复，**尚未完成 fresh 正常 CLI 复跑**。

注意：此前 W1 曾手动重跑 reading 写入失败 workspace，不能再把该目录所有产物当作同一次完整调度的证据。恢复时使用新的 workspace，保留旧失败目录。

## 尚未完成的工作与恢复优先级

### P0：正常全链与独立验证的收敛

1. 在新的 workspace 运行上述真实双 UID 正常 CLI，解决剩余阶段契约问题，产出 TTS JSONL/TextGrid，并达到独立 verifier 的 `integrity_ok=true`。开发配置预期仍不满足 release gate。
2. 使用三 UID mixed 配置验证真实双语言 runs、多个日语 token、逐 token reading、mora 关系、两套原始 TextGrid 和字典 provenance；不能用单 token 或全日语样例冒充 mixed 正例。
3. 核对 bridge 与 verifier 的实际外部证据字段：reading/analysis/reconstruction/alias/semantic 路径和 hash、按 UID/token/candidate 精确 join、每个 MFA run 的 archive/inventory/dictionary/TextGrid/crop offset；避免 producer 自填副本互相证明。
4. 核对 semantic/mora 关系逐 token 对证，只有 pronunciation、target native sequence、raw MFA 顺序一致才建立关系；不能按全句数组索引或 native label 集合蒙混通过。English graph 不应被全局 Japanese mora 约束误拒绝。
5. 核对 UID/unit/run 分区守恒，单 UID unresolved 不应让整批无账本崩溃；全拒收不允许 COMPLETE。

### P1：恢复与缓存身份

core 在用户停止时仍在修改这些项目，必须读取实际文件和定向测试确认，不可假定已完成：

- Julius config/资产/实现变化不污染 production MFA/TTS identity；root receipt 使用 production 配置及供应链快照。
- frontend analysis 与 reconstruction 分离依赖，避免第一次 resume 因 reading 依赖被重写而无故重跑。
- runtime identity 保留实际调用的 venv Python 路径、symlink target、包环境；最近审查时只记录包名／版本／文件名列表，仍需确认是否已覆盖 RECORD/模块内容漂移。
- provider 嵌套模型/runtime 相对路径、所需 input 不存在时的拒绝、`ja_en_stage_inputs.py` 实现 hash。
- 模板补齐真实 handler 消费的 provider runtime/revision/device 与 MFA archive hash 等字段。
- fresh 完整运行后验证 resume 不重算且产物 hash 稳定；变更输入/模型/flags/mapping/环境时拒绝旧缓存；Julius 单独开关不改变生产产物。

### P2：发布门禁与最终回归

- canary 结果应从实际 pipeline readings/route/seams/partition 生成并绑定外部 gold，不能仅接受手写 PASS 或 producer 声明值；需完整正常配置路径。
- reading 60 条（consistent/ambiguous 各 30），pure 日英各 10，mixed 四桶各 10，总接受至少 34、每桶至少 8，MAE ≤80 ms、P95 ≤160 ms、route/clipping/overlap/false-COMPLETE 为零。
- 最终统一运行全量 pytest、编译检查、diff 检查、受保护源文件 hash 检查，记录准确版本与结果。
- 根据最终实现修订操作文档；保留开发完整性通过与生产 release ready 的区别。

## 外部生产条件仍未具备

没有用户游戏 corpus 路径／owner 人工读音政策，没有人工 held-out gold，Whisper 与 Reazon baseline runtime/model 尚未实际运行，资源许可证据尚未由 owner 审核，完整 native inventory 语言学审查和 mixed 质量 gate 未完成。Julius binary/model/dictionary 资产未准备，但它是可选诊断，不应阻断正常生产分支。

这些条件会阻断生产发布；它们不能代替上面 P0/P1 的工程联调。当前正确结论是：**主要模块已实现，真实组件 smoke 已通过，整链集成与最终验收未完成，用户主动结束本次实现。**

## 续接索引与协作历史

长期日志：[progress.md](/home/user/ja-en-task-artifacts/progress.md)；验收索引：[acceptance-tracker.json](/home/user/ja-en-task-artifacts/acceptance-tracker.json)；环境记录：[environment-handoff.md](/home/user/ja-en-task-artifacts/environment-handoff.md)。这些外部文件记录历史过程，若与本文最新停止状态冲突，以本文和保存的源文件为准。

原角色所有权：W0 core/schema/config；W1 source/audio/supply-chain/ASR；W2 frontend/text/semantic；W3 anchors/MFA/merge/bridge；W4 export/verifier/gates/docs；W5 Julius。用户此前已批准 Sol 容量不足时直接按已有方案，由 Terra 协调、Luna 实现。当前用户已要求结束实现，后续必须等待新的继续指令。
