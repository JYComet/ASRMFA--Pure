# English/RIA MFA 严格管线问题分析与实施交接方案

## 元数据

- 生成时间（UTC）：2026-08-07T07:54:58Z
- 仓库路径：/mnt/local_E/MFA_Pause/repo
- revision：854ea5a7810954bd092276ab13759ca58effbde9
- 分支：main
- 工作树：存在既有未提交修改；本方案未覆盖、删除或重置这些修改
- 任务 slug：english-mfa-strict-pipeline
- 交接文件：handoffs/20260807T075458Z-english-mfa-strict-pipeline.md
- 规划路由：根代理已显式接受 gpt-5.6-sol、reasoning_effort=high、agent_type=default、无继承会话分叉的启动请求

## 目标

为后续执行窗口提供一份可直接实施的方案，闭环 REGRESSION_ARCHIVE.md 与 ENGLISH_MFA_TASK_HANDOFF.md 暴露的 RIA/English MFA 管线问题。重点是：

1. 固定 reference-only English CTC 的内容权威、时间轴和 provenance；
2. 消除 CTC、MFA、postprocess、report、filtered/output 与发布之间的集合和事务漏洞；
3. 将 RIA 旧批次的语义完整性问题与 English strict v4 的 fresh rerun 问题分层处理；
4. 在任何生产 GPU、MFA canary 或版本化发布前，建立独立、可故障注入的验收门。

本交接只规划，不实现代码，不运行生产任务，不修改已有源文件。

## 背景与当前行为

ENGLISH_MFA_TASK_HANDOFF.md 已将当前任务判定为 NO-GO，明确禁止新的 English CTC 声学 canary、全量 GPU rerun 和 MFA 生产任务（ENGLISH_MFA_TASK_HANDOFF.md:1-5）。English strict v4 的权威输入是 54,000 个 WAV、53,998 个 TXT、53,998 个 WAV/TXT 交集，另有 2 个缺参考文本 stem；配置中的 ready evidence hash 仍为占位值，不能进入完整管线（ENGLISH_MFA_TASK_HANDOFF.md:62-84）。

参考文本模式的产品不变量已经冻结：参考文本决定 CJK、英文、NVV、标点和句首 sp1；ASR/CTC 只提供时间边界；参考与音频不匹配时硬失败，不得回退到 ASR 内容（ENGLISH_MFA_TASK_HANDOFF.md:88-117）。

RIA 旧批次暴露了另一条失败链：cn2an 曾把 rui4 变为 rui四，导致 lab 大量 OOV，MFA 输出 unknown/spn，后处理又把 unknown 排除，最终出现 0 pinyin 与正数 CJK 的语义坍缩（REGRESSION_ARCHIVE.md:6148-6189、6235-6284）。因此“进程返回 0”“TextGrid 文件存在”或“词面被 CTC 回填”都不能单独证明声学对齐有效。

当前工作树已部分加入修复草案和验证器，但这些改动尚未形成可放行的整体契约。最新档案确认 Cases 97、98、100、101、102 已有代码或单元回归，Case 99 仍无实现；Cases 78、79、81、83、84、85 等仍未闭环或未验证（REGRESSION_ARCHIVE.md:6416-6655、6571-6618、6910-6978、7218-7391）。

## 事实（Facts）

1. 仓库规则要求逻辑性 bug 修复写入回归档案，并要求记录现象、根因链、函数/行号和可复现验证（CLAUDE.md:1-25）。
2. 工作树已有用户修改，包含 REGRESSION_ARCHIVE.md、两个配置和多个 scripts 文件，同时存在新增的准备器、验证器和状态文档；这些改动必须保留。
3. ctc_prealign.py 已在自由解码 clone 上屏蔽 reference-only 的 NVV ID，并让 forced alignment 使用干净 logits（scripts/ctc_prealign.py:160-177、246-255）。
4. ctc_prealign.py 当前仍用 encoder frame 轴计算推理中间 duration_s，再在写盘阶段读取 WAV duration 并 clamp lexical token；两种时间轴在同一流程中并存（scripts/ctc_prealign.py:296-305、353-397、550-564、1568-1741）。
5. CTC writer 已将 query frame 过滤后的 blank run 保存为 speech 坐标，并在 pause 写盘时做边界检查（scripts/ctc_prealign.py:437-457、511-545）。
6. CTC manifest 在 normalize 之后由最终 tokens 重建，这是 Case 97 的已实施方向；但该函数只重建 manifest，不记录实际模型树、运行 argv 或成功 receipt（scripts/ctc_prealign.py:883-903、2046-2067）。
7. all-GPU 路径会先检查 shard namespace、manifest、summary、marker 和 stem union，但随后仍把 shard 文件移动到父目录，再进行 bundle 校验和最终 manifest 重建；移动和字典追加不是可回滚事务（scripts/ctc_prealign.py:1246-1317、1319-1427）。
8. v4 prepare 和独立 verifier 已检查 reference sidecar、六件套、WAV domain、manifest、source inventory 和 copy hash，但没有 model tree digest、实际 CTC run receipt 或 shard receipt 的绑定（scripts/prepare_hecheng_english_ctc_ready.py:811-836、894-915；scripts/verify_hecheng_english_ctc_ready_v4.py:177-239）。
9. v4 timing 校验对 lexical token 强制 start/end 各自单调，punct 则不做序列单调比较；这符合允许 token/punct overlap 的方向，但当前测试没有覆盖嵌套 punct overlap 的反例（scripts/prepare_hecheng_english_ctc_ready.py:838-848；REGRESSION_ARCHIVE.md:7133-7140）。
10. run_pipeline.py 的 strict CTC import 通过独立 verifier 后才复制到 fresh workspace，并对导入副本做 hash、普通文件和 inode alias 检查（scripts/run_pipeline.py:2779-2878）。
11. MFA sharded path 只检查输出文件中是否包含 words 和 phones 字符串，未验证完整 TextGrid grammar、tier 唯一性、interval 合法性、NaN、倒置或音频 domain（scripts/run_pipeline.py:1506-1533；REGRESSION_ARCHIVE.md:6571-6592）。
12. English parser parse_en_textgrid 同样按文本行和 tier 顺序宽松抽取，tier 少于两个时仅返回空 words，不能成为 strict parser（scripts/align_english_mfa.py:953-1031）。
13. postprocess 已以不可变参考文本执行 CJK coverage，并将 unknown_source 纳入 hard integrity reasons；其 unknown 分类现在只承认明确占位符，不把裸词 unk 自动视为 unknown（scripts/postprocess_textgrids.py:1906-1965、5041-5117；scripts/pipeline_utils.py:947-991）。
14. postprocess 入口已检查 expected、lab、audio、aligned 的集合，并在 hard integrity 失败时返回非零；但 RIA 共享 filtered/output 和 NAS 固定目录的历史污染仍需由 fresh run 与版本化发布门隔离（scripts/run_pipeline.py:1867-1913、3936-3961；REGRESSION_ARCHIVE.md:6416-6449）。
15. configs/hecheng_english_mfa.yaml 已禁止 padding、cache 和 direct staging，并把 ready evidence hash 留作 fail-closed 门；配置仍依赖 prepare/finalize 后的真实证据（configs/hecheng_english_mfa.yaml:1-45）。

## 假设（Assumptions）

1. 本任务的执行目标沿用交接文档冻结的 reference-only English 语义，不重新引入 ASR-added NVV。
2. 旧 RIA 目录、旧 English CTC、旧 aligned 和旧 NAS 输出均视为只读问题现场；恢复只在新 workspace/run root 和新版本发布目录进行。
3. “允许 overlap”只针对 CTC token/punct 的跨容器时间语义；同一 TextGrid tier 内仍要求区间正时长、有限、单调且在音频轴内。
4. 具体 production source、model、MFA Python 和 GPU 资源在执行窗口仍需从磁盘和宿主重新确认，不能只依赖文档中的历史路径。
5. 当前已实施的代码只视为候选修复，除非通过本方案中的 fault tests、独立 verifier 和 fresh canary。

## 决策（Decisions）

1. 先完成无 GPU 的契约和故障回归，再做只读 inventory，再创建全新 run root，最后才允许 canary；任何硬门失败都停止。
2. 将 CTC writer、prepare/finalize、独立 verifier 和 pipeline import 视为一个版本化 schema，禁止用宽松兼容逻辑掩盖结构未知或 provenance 缺失。
3. 模型身份采用 exact regular-file tree、文件 size/hash 和稳定 tree digest；只记录路径或 mtime 不足以证明运行身份。
4. all-GPU 合并采用 staging-first：先在 shard 和内存中完成完整验证，再复制到新的父 staging，验证通过后按 marker-last 规则一次发布。
5. single MFA 与 sharded MFA 共用同一严格 TextGrid validator 和结构化失败语义；不能保留字符串搜索作为成功条件。
6. RIA 的 output、filtered、report、logs 和 publish target 使用 run-id 隔离；旧固定目录不清理、不增量覆盖、不作为新结果分母。
7. 任何需要改变 reference authority、公共 evidence schema 或发布事务规则的发现，都回到 root 和 Sol high 重新批准。

## 未决问题（Open Questions）

1. ENGLISH_MFA_TASK_HANDOFF.md 的快照称宿主没有 Luna 路由（ENGLISH_MFA_TASK_HANDOFF.md:30-32），而当前根代理可见的模型覆盖目录列出 gpt-5.6-luna。执行责任人必须在 Gate 0 重新记录实际可用 route；若批准的 Build/Inspect 路由不可用，不得替换模型。
2. 036000 缺失权威参考文本是否补齐尚未由业务确认；在确认前只能排除，不得用 ASR fallback 混入 53,998 个 reference-only stem（REGRESSION_ARCHIVE.md:6657-6685、7445-7451）。
3. Case 93–95 旧 CTC 不再作为 v4 ready 输入，但是否需要单独做历史恢复仍待业务决定；若恢复，必须遵守独立的 grammar、时间轴和源音频证明，不能复用旧格式宽松 parser（REGRESSION_ARCHIVE.md:6927-7070）。
4. Case 78/79 的版本化发布与 tone_mapping 归档仍需确认最终交付目录命名、保留周期和发布授权；未授权时使用 no-output-staging。

## 范围与约束

### In scope

- reference-only CTC 内容隔离、reference sidecar、六件套和空 punct sidecar；
- WAV authoritative axis、query-frame pause、token/punct/VAD endpoint 规则；
- CTC final manifest、marker、model tree、run receipt、shard receipt；
- all-GPU preflight、staging、父级 marker-last 和失败现场保留；
- MFA strict parser、single/sharded 统一验证和结构化进程失败；
- postprocess 的 reference CJK/NVV/punctuation/sp1 integrity、RIA 三方原子同步；
- expected/passed/filtered/report 的集合守恒、版本化 output 和 publish manifest；
- 无 GPU fixture、独立 verifier、compile、scoped diff 和 canary 验收方案。

### Out of scope

- 不修改既有工作树中与本任务无关的代码或配置；
- 不删除旧 CTC、旧 MFA、旧 output、旧 filtered 或 NAS 现场；
- 不把 filtered 内容质量在本轮提升为新的业务验收目标；
- 不启动真实 GPU/MFA、全量 rerun、NAS 发布或替换配置中的 evidence hash；
- 不重新设计参考文本作为内容权威的产品决策。

### 约束

- 保留当前所有未提交改动；执行前后均需记录 scoped diff。
- 新 run root、workspace、output、filtered、report 和 logs 必须不存在；失败 root 保留，重试使用新 root。
- strict v4 禁止 post-CTC padding、cache reuse、force、overwrite 和固定目录增量发布。
- 回归档案中新增逻辑性 bug 时要追加完整 case 证据；已有 case 只补充同一根因的验证记录。

## 受影响文件与符号

| 文件 | 关键符号/区域 | 证据与作用 |
|---|---|---|
| scripts/ctc_prealign.py | _free_decode_logits、make_patched_inference | reference-only decoder mask、forced alignment 和 query-frame 坐标（160-457） |
| scripts/ctc_prealign.py | write_textgrid、_clamp_words_to_wav_axis | TextGrid domain、pause/token endpoint（468-564） |
| scripts/ctc_prealign.py | main 的 all-GPU 分支 | shard 预检、移动、合并和 marker（1104-1429） |
| scripts/ctc_prealign.py | _rebuild_final_manifest、处理循环 | final manifest、WAV duration、tokens/punct 写盘（875-903、1568-2067） |
| scripts/prepare_hecheng_english_ctc_ready.py | render_rerun_command、prepare、_v4_load、finalize | fresh root、command、copy、bundle、evidence（811-915） |
| scripts/verify_hecheng_english_ctc_ready_v4.py | _reference_projection、timing、tgparse、validate_bundle、verify | 独立内容、时间、TextGrid、inventory 和 namespace 证明（26-239） |
| scripts/pipeline_utils.py | load_ctc_token_entries、validate_ctc_transcript_bundle | lab/tokens/TextGrid 三方词序和 token 时间契约（1023-1154） |
| scripts/pipeline_utils.py | is_unknown_token、is_word_like、is_punct | unknown/真实英文/标点分类（947-991） |
| scripts/run_pipeline.py | _skip_if_ctc_normalized、_run_mfa_sharded | marker gate、MFA shard 验证和合并（852-896、1295-1594） |
| scripts/run_pipeline.py | _step_link_ctc_strict、strict_stage_denominator_issues | ready import、copy identity、阶段分母（2646-2878） |
| scripts/align_english_mfa.py | parse_en_textgrid、run_en_mfa、strict ledger | English MFA 输出解析、进程结果和 provenance（877-1031、1204-1418） |
| scripts/postprocess_textgrids.py | assess_reference_coverage、process_one、main | RIA/English lexical integrity、unknown source、report 和 hard failure（1906-1965、5041-5117、6171-6188、7240-7301） |
| configs/hecheng_english_mfa.yaml | strict v4 paths and gates | fresh CTC ready、no padding/cache/staging、evidence pin（1-45） |
| configs/hecheng_ria_0805.yaml | legacy RIA mode | 固定 workspace/output 和可导致旧目录污染的运行语义（19-57、89-98） |
| scripts/verify_prepare_hecheng_english_ctc_ready.py | v4_main 和 fixture tests | 当前回归覆盖与缺失覆盖的入口（183-252） |
| REGRESSION_ARCHIVE.md | Cases 72-102 | 已知根因、状态、停止条件和验收约束（6148-7391） |
| ENGLISH_MFA_TASK_HANDOFF.md | sections 2-10 | 已批准产品不变量、串行包、fault tests、Gate 0-12（88-645） |

## 编号要求

R1. 维持 reference-only authority：English rerun 命令恰有一次 --no-nvv，每个 child/shard 继承；自由 ASR 只能进入 diagnostic，不得进入 required sidecar。

R2. 固定七类 required artifact 的来源：_ref.txt、_text_raw.txt、_text_cn.txt、.lab、tokens、TextGrid words、punct 必须与 canonical reference 在词面、NVV 数量顺序、标点顺序和 lexical projection 上一致；无标点 stem 必须有合法 [] punct JSON。

R3. 统一 CTC 时间轴：每个 stem 用 run-local authoritative WAV header duration；token、punct、pause、VAD endpoint 分别验证有限、正时长、domain 和约定的单调/overlap；允许的尾部量化裁剪必须重算字段且不得产生零长。

R4. 将 final manifest/marker 变成最终 bundle 的事务证据：normalize 完成且最终六件套通过后才发布；marker 必须绑定最终 manifest digest 和精确 stem count。

R5. 实施 Case 99 provenance：prepare 冻结模型 exact regular-file tree、词典、ASR Python、规范化 argv 和输入 stem digest；实际 CTC 成功后写原子 receipt；finalize/verifier 重算并交叉核对。

R6. 将 all-GPU 合并改为真正 staging-first 事务：任何 shard 缺失、重复、额外、坏 manifest、坏 marker、目标碰撞或父级验证失败都非零；父级不得留下 final marker 或可误认的最终 manifest。

R7. 将 MFA 输出验证统一为 strict parser：single 与 sharded 共享完整 long TextGrid grammar、唯一 words/phones tier、finite positive monotonic in-domain intervals、expected/produced stem exact set 和结构化失败记录。

R8. 收敛进程异常：Popen OSError、timeout、signal exit、non-zero、日志不可读和输出不完整都进入持久化 failure manifest；失败现场保留且禁止 postprocess。

R9. 保护 RIA/English postprocess 语义：reference CJK 与 pinyin 一一对应，合法 NVV、标点、sp1 不计入 CJK denominator，MFA source unknown/spn 不能伪装为声学成功；裸词 unk 不得被误判为占位符。

R10. 实施 RIA 三方原子同步：normalizer/merge 同时更新 lab、tokens、TextGrid 或全部不提交；失败时旧 marker 失效，重新验证后才写新 marker。

R11. 固定 run-id 集合契约：expected = passed ∪ filtered，二者交集为空，report 每 stem 恰一行；output、filtered、tone_mapping、logs 和 publish manifest 必须来自同一 run，旧目录不得参与分母。

R12. 建立从无 GPU 到 canary/full 的门禁：所有专项 fault tests、独立 verifier、compile、scoped diff、inventory 和 evidence pin 通过前不得运行生产任务；canary 必须使用新 root 且包含冻结的代表性 stem。

## 有序实施计划

### 0. Gate 0：重新盘点并冻结边界

所有者：root。依赖：无。

只读记录 git revision、branch、status、现有 diff、进程、配置、模型路径和实际路由。确认旧目录只读、strict v4 目标 root 不存在；重读两个输入文档并冻结本方案 R1-R12。若发现 reference authority、公共 schema 或发布事务需要改变，停止并返回 Sol high/root 决策。

### 1. Reference-only CTC bundle

所有者：CTC/prepare 实施者。依赖：0。

在 ctc_prealign.py 中将 reference_only 明确贯穿 decoder、writer、manifest 和 child argv；保留非 reference-only 模式的 ASR-added NVV。写盘前生成 canonical reference sidecars、reference-derived lab/tokens/TextGrid words/punct，并将 ASR text 仅放入 diagnostic 字段。为每个 stem 统一写空 punct JSON，禁止在异常路径留下半套 artifact。

### 2. CTC 时间轴、manifest 和 provenance

所有者：CTC/证据实施者。依赖：1。

抽取独立的 WAV duration、token/punct/pause/VAD validator；将 lexical start 与 end、毫秒序列化、WAV domain 和允许 overlap 分开定义。将 normalizer 后 final bundle、manifest、marker、model tree、dictionary、argv、Python、input/success stem set 收敛到一次成功事务。receipt 在所有 stem 成功后原子发布，失败只保留 failure receipt/日志，不生成 success marker。

### 3. All-GPU staging transaction

所有者：CTC 合并实施者。依赖：2。

每个 shard 生成精确 namespace、manifest、summary、marker、receipt；父级先在内存中验证 artifact 与 stem 一一对应、shard 集互斥且并集精确，再复制到新的 parent staging。所有最终 bundle、manifest、summary、dict identity 和 namespace 验证成功后，原子写 manifest/summary，最后写 parent marker。任何异常不清理 shard，不写 parent success metadata，不允许续跑旧根。

### 4. Strict MFA parser 与进程契约

所有者：MFA 实施者。依赖：3。

在共享模块定义严格 long TextGrid parser/validator，显式检查 grammar、tier 名称/数量/顺序、interval 有限性、正时长、单调性、WAV domain、expected lexical words 和 phones。run_pipeline.py 的单进程/分片路径、align_english_mfa.py 的 English path 统一调用同一语义 validator；保留每 shard log、argv、return code、timeout、signal/exception、expected/produced/rejected stems。

### 5. Postprocess 与 RIA atomic sync

所有者：postprocess/质量实施者。依赖：1、4。

继续以 original/ref reference 为权威；为 CJK projection、NVV、punctuation、sp1、unknown/spn、English phone provenance 建立独立审计。normalizer 或 ria merge 在临时副本中同时改 lab、tokens、TextGrid，完成三方验证后替换；失败清除旧 marker。tone_mapping 写入当前 run output，不能写仓库默认 output。

### 6. 集合、版本化发布和报告

所有者：pipeline/release 实施者。依赖：4、5。

为每次运行生成唯一 run-id 目录，清空语义改为 fresh target 门禁而非删除旧文件。执行 expected/passed/filtered/report exact conservation、tone_mapping/report/log manifest hash 校验；发布只接受 strict_ok 成功且目标版本目录不存在的目录，发布后从目标重读并核对 manifest。未获业务授权保持 no-output-staging。

### 7. 无 GPU 回归与独立审计

所有者：验证实施者。依赖：1-6 对应代码完成。

增加 reference-only content mutation、NVV/punct/sp1、nested overlap、WAV 1.00/9.44 秒、query boundary、model tree tamper、receipt drift、all-GPU failure、MFA grammar/process failure、RIA three-way sync、unknown/unk、集合污染和 tone_mapping path fixture。prepare 与 standalone verifier 使用不共享 parser 的交叉实现。

### 8. Fresh inventory、canary 和 full run

所有者：root + 运行责任人。依赖：7 全部通过，Sol/root 明确 GO。

先执行 authoritative inventory，确认 54,000/53,998/53,998/2/0 和具体 missing stems；再使用不存在的新 run root prepare、CTC rerun、finalize、独立 verifier。canary 仅选择代表性 stems，必须包含纯英文、CJK/English、reference NVV、标点、sp1、ria、数字、非 60ms WAV、历史 0 pinyin 风险样本。canary 全部硬门通过后才允许 full run；full run 仍不使用 force/overwrite。

## 要求到验收追踪

| 要求 | 客观验收标准 | 验证步骤/命令 |
|---|---|---|
| R1 | command 中 --no-nvv 恰一次；child argv 全继承；ASR 改写不改变 required artifact | verify_prepare；reference-only fault test；检查 rerun receipt argv |
| R2 | 每个 expected stem 的 lab/token/TextGrid lexical 序列一致；reference NVV/punct 顺序相同；无标点 JSON 为 [] | independent v4 verifier；reference authority suite；bundle namespace 检查 |
| R3 | TG/tier/manifest domain 等于 WAV header；token/punct/pause endpoint 合法；nested overlap 不误拒 | CTC timing fixture；verify_hecheng_english_ctc_ready_v4.py；1.00/9.44 秒 fixture |
| R4 | marker digest 等于最终 manifest；normalizer 改词数后 manifest 与最终 tokens 相同；失败无 success marker | CTC final manifest regression；marker tamper negative test；fresh-output test |
| R5 | 替换同路径 model file、symlink、extra file、argv/dict/stem receipt drift 均在 MFA 前非零 | model tree/receipt fault suite；finalize 和 standalone verifier |
| R6 | duplicate/missing/extra shard、目标碰撞、坏 marker 或父验证失败均非零，父根无 final marker/manifest | all-GPU synthetic transaction suite |
| R7 | 截断/重复 tier、NaN、倒置、越 WAV、缺 phones 的 single/sharded MFA 均失败 | strict TextGrid parser fixture；run_pipeline align test；English parser test |
| R8 | OSError、timeout、signal、non-zero 和 incomplete output 写入结构化 manifest，失败目录保留 | subprocess fault injection；检查 logs、return_code、failure manifest |
| R9 | reference CJK 缺 pinyin、CJK sequence mismatch、MFA source unknown/spn 不得 status=ok；裸 unk 真实英文不被过滤 | postprocess fixture；audit_strict_ok.py；unknown/unk context test |
| R10 | ria 合并后 lab/tokens/TextGrid 三方词序和 token 数一致；任一写入失败不更新 marker | three-way atomic sync fixture；marker lifecycle regression |
| R11 | passed ∩ filtered 为空，union 等于 expected，report 每 stem 一行，tone_mapping 位于 run output | strict denominator audit；publish manifest verify；stale file fixture |
| R12 | 无 GPU gates 全部返回 0 后才创建 canary；canary/full 使用 fresh roots，无 evidence placeholder | compile/diff/test suite；inventory、prepare/finalize/verifier gate；root GO record |

## 可观察验收标准

- English ready evidence 的 schema、state、independent verifier signature、taxonomy hash 和 evidence hash 均为真实值，不再有占位值。
- reference-only required artifact 中不存在 ASR-only CJK、英文、NVV 或标点；reference NVV 和标点只出现一次且顺序一致。
- 每个 CTC bundle 的 TextGrid、tier、tokens、punct 和 manifest 使用对应 WAV header 轴；所有文件为普通文件，namespace 无 extra/missing。
- 每个模型执行 receipt 可回溯到实际 Python、argv、model tree digest、词典 digest、输入 stem digest 和成功 stem digest。
- all-GPU 父级只有在完整验证后才有 final marker；任何失败根均无可被下游识别为完成的 parent success metadata。
- MFA 解析失败、缺 stem、进程异常或 source unknown 均不能流入 postprocess。
- postprocess 报告对 expected 每 stem 恰一行，passed/filtered 互斥且并集守恒；hard integrity 失败返回非零。
- RIA/English 的 tone_mapping、output、filtered、logs、report 和 manifest 可由同一 run id 关联，旧固定目录不作为本次结果。

## 验证命令与预期信号

执行窗口的 shell 必须先确认路径和解释器；以下命令均为无 GPU 或只读门禁，不能替代 canary：

~~~bash
cd /mnt/local_E/MFA_Pause/repo
git diff --check
python -m compileall -q scripts check_ipa_mapping.py verify_risks.py
python scripts/verify_prepare_hecheng_english_ctc_ready.py
python scripts/verify_strict_ctc_ready_import.py --verbose
python scripts/verify_reference_authority.py
python scripts/verify_reference_only_ctc.py
python scripts/verify_strict_ok.py
python scripts/verify_tier_discontinuity.py
~~~

预期：所有命令返回 0；每个 fault test 明确覆盖正例和负例；没有未解释 warning；如果任何命令失败，保持 NO-GO。

只读 inventory：

~~~bash
python scripts/prepare_hecheng_english_ctc_ready.py inspect \
  --source-dir /mnt/Raw/新版合成英文数据 \
  --require-expected-counts \
  --expected-wavs 54000 \
  --expected-txts 53998 \
  --expected-authoritative 53998 \
  --expected-missing-refs 2 \
  --expected-txt-only 0
~~~

预期：计数精确匹配，missing-reference 列表与冻结清单相同；该命令不创建 run root。

fresh prepare/finalize/verifier 只在上述门通过后使用新 root。命令参数必须从 prepare_manifest 的 exact rerun command 读取，禁止手工添加 overwrite、force 或旧 root 复用。finalize 成功后独立执行：

~~~bash
python scripts/verify_hecheng_english_ctc_ready_v4.py \
  --run-root /mnt/nvme3/mfa_runs/hecheng_english/20260807T075458Z_strict_v4_1 \
  --source-dir /mnt/Raw/新版合成英文数据 \
  --dictionary-source /mnt/local_E/MFA_Pause/repo/dict/mfa_ipa.dict
~~~

预期：打印 v4 ready evidence verified；任何 evidence、model tree、manifest、namespace、authority substitution 或时间轴负例均返回 1。

## 注意事项与不变量

1. 不以 verifier 放宽容差修复 writer 的错误；WAV domain、seconds/milliseconds 序列化和 lexical start 容差必须分开。
2. 不从 tokens end 推写旧 TextGrid end；跨容器只比较已冻结的词面、顺序和 start，单容器内分别验证自身 end。
3. 不把合法 NVV、punct、sp1、英文排除 CJK denominator 的事实误报为 0 pinyin；也不允许这些合法标签掩盖真实缺 CJK。
4. --no-nvv 的 decoder mask 只能作用于自由 decode clone；reference NVV 的 forced alignment 必须继续使用原 logits 和 reference target。
5. marker 是优化信息，不是 validation 替代品；任何直接 step=align 或 marker skip 都要重新做 bundle quick validation。
6. hardlink/symlink 可能使 ready source 被 pad/normalize 穿透修改；strict import 使用普通 copy 并验证 inode 不 alias。
7. 失败 root 不清理、不续跑、不手工拼接；重试必须使用新的版本化 root。
8. RIA 旧结果只能作为证据，不得作为新输出分母或发布源。

## 风险与回滚

| 风险 | 影响 | 缓解/回滚 |
|---|---|---|
| 模型树或词典在运行中漂移 | 时间边界不可追溯，ready provenance 无效 | 运行前冻结 tree，receipt 不一致即废弃 root，用新 root 重跑 |
| all-GPU 父合并中断 | 混合或不完整 bundle | staging-first，失败保留 shard，删除资格由人工审计决定；不续跑，换新 root |
| strict parser 误拒合法 TextGrid | canary 阻断 | 先用标准 grammar fixture 校准；未知 grammar 归 rerun/人工决策，不放宽到字符串搜索 |
| RIA 三方 merge 部分写入 | lab/tokens/TextGrid 不同步 | 临时文件和统一 commit；失败清除 marker，隔离副本重建 |
| output/filtered stale 污染 | 通过率与发布集合失真 | run-id 独立目录，exact manifest，发布后重新读回验证 |
| reference 缺失或源文本漂移 | ASR 内容污染权威批次 | 缺失 stem 排除；源 hash 变化立即失败，不 fallback |

回滚策略：不回滚用户既有工作树。只废弃失败的新增 run root、workspace staging 或未发布的 evidence；保留失败日志和 manifest 作为审计材料，下一次从新的空 root 开始。旧 NAS/旧 workspace 不删除。

## 未解决阻塞项

| 阻塞项 | 证据 | 影响 | 责任人 | 决策路径 |
|---|---|---|---|---|
| Case 99 model tree/receipt 未实现 | REGRESSION_ARCHIVE.md:7261-7295 | 无法证明真实模型身份，禁止 prepare/canary | CTC/证据实施者 | 先实现并通过负例，再由 Sol/root GO |
| Cases 76/83 strict MFA parser 未闭环 | REGRESSION_ARCHIVE.md:6351-6386、6571-6592；scripts/run_pipeline.py:1506-1533 | 缺失或坏 TextGrid 可能进入 postprocess | MFA 实施者 | 统一 parser 后执行 single/sharded fault suite |
| Case 78/79 发布与 tone_mapping 未完成真实验证 | REGRESSION_ARCHIVE.md:6416-6468 | 固定目录可能混入陈旧输出，映射可能不随 run 交付 | pipeline/release 实施者 | 业务确认发布授权，root 执行版本化验收 |
| RIA Case 81 三方同步未闭环 | REGRESSION_ARCHIVE.md:6508-6527 | lab/tokens/TextGrid 词序和 marker 不一致 | postprocess/CTC 实施者 | atomic sync fixture 通过后再恢复 RIA |
| 036000 权威参考缺失 | REGRESSION_ARCHIVE.md:6657-6685 | 53,998 reference denominator 不能合法包含该 stem | 业务数据负责人 | 补 txt 或确认显式排除 |
| 历史 93-95 是否恢复 | REGRESSION_ARCHIVE.md:6927-7070 | 可能扩大范围并把格式转换误当声学修复 | root/业务负责人 | 明确“只做 v4 fresh rerun”或另立恢复任务 |

## 执行清单

- [ ] 记录新窗口 UTC、revision、branch、status、模型路由和运行进程。
- [ ] 阅读 CLAUDE.md、两个任务基线并确认所有既有修改保留。
- [ ] 完成 R1-R4 reference-only、时间轴、manifest、marker 事务。
- [ ] 完成 R5 model tree、receipt、dictionary/argv/stem binding。
- [ ] 完成 R6 all-GPU staging-first 和 marker-last。
- [ ] 完成 R7-R8 strict MFA parser、日志和失败 manifest。
- [ ] 完成 R9-R10 postprocess integrity、unknown/unk 和 RIA atomic sync。
- [ ] 完成 R11 run-id、report、tone_mapping、publish manifest 集合守恒。
- [ ] 新增并通过所有 R1-R12 fault tests；执行 compile 和 scoped diff check。
- [ ] 重新执行只读 inventory，确认精确计数和 missing 清单。
- [ ] 创建不存在的新 prepare root；禁止复用失败 root。
- [ ] 完成 CTC rerun、finalize、独立 verifier，并将真实 evidence hash 写入配置前再次复核。
- [ ] 由 Sol/root 明确 GO 后执行代表性 canary。
- [ ] canary 通过后重新确认 full run GO；全量结果做独立审计。
- [ ] 未获发布授权时保持 no-output-staging；授权后只发布新版本目录。

## Readiness 决定与门结果

当前决定：NO-GO，尚未达到实现或生产运行就绪。

| Gate | 结果 | 依据 |
|---|---|---|
| Sol high route | PASS | 根代理显式启动请求被接受，模型和 high effort 在根可见 schema 中列出 |
| 仓库/规则发现 | PASS | CLAUDE.md、README、两个任务基线和受影响代码已读取 |
| 证据充分 | PASS（规划层） | Cases 72-102、当前代码符号、配置和验证脚本已交叉检查 |
| 现有工作保护 | PASS | 未编辑源代码、测试、配置、原交接文档或 git 状态 |
| Case 99 | FAIL | 无 model tree digest、CTC run receipt 和 shard receipt |
| Cases 76/83 | FAIL | MFA 输出检查仍是字符串包含，缺 strict shared parser |
| Cases 78/79/81 | FAIL | 版本化发布、tone_mapping 随 run、RIA 三方原子同步未完成真实验收 |
| 无 GPU 回归 | FAIL | 当前测试未覆盖全部要求的 nested punct、namespace、receipt、MFA grammar/process negative cases |
| fresh inventory/evidence pin | NOT RUN | 生产源、fresh root、GPU 和 evidence pin 均按文档冻结 |
| canary/full | BLOCKED | 必须等待上述 FAIL 项闭环和 root GO |

结论：本文件已具备下一窗口实施条件，但当前仓库不具备生产运行条件。

## 下一窗口启动指令

在 /mnt/local_E/MFA_Pause/repo 启动新窗口，先完整阅读：

handoffs/20260807T075458Z-english-mfa-strict-pipeline.md

随后只读核对当前 git status、revision、路由和既有 diff，从 Gate 0 开始。先执行未完成的 R5/R7 相关无 GPU 工作；不得启动生产 CTC/MFA，不得修改固定输出目录，不得替换 configs/hecheng_english_mfa.yaml 中的 evidence 占位值。只有完成所有门、刷新证据并取得明确 GO 后，才进入 fresh canary。
