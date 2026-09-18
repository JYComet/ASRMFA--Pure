# English/RIA MFA 严格管线任务交接文档

> 快照日期：2026-08-07  
> 项目根目录：`/mnt/local_E/MFA_Pause/repo`  
> 当前结论：**NO-GO，禁止启动新的 English CTC 声学 canary、全量 GPU rerun 或 MFA 生产任务。**

## 新对话窗口直接使用的指令

新对话无需重新分析项目目标和整体管线。建议直接发送：

```text
先完整阅读 /mnt/local_E/MFA_Pause/repo/ENGLISH_MFA_TASK_HANDOFF.md，
将其作为已经批准的 TASK_BRIEF 和问题基线继续执行。
不要重新分析任务目标；先核对当前工作树和实际可用模型路由，
然后从文档“执行顺序”中第一个未完成的门开始。
不得覆盖、删除或重置现有未提交修改，不得在 NO-GO 状态启动生产任务。
```

如果继续使用 `$route-coding-agents`，已经批准的架构顺序是：

```text
Sol high Strategy（已完成 v6）
→ root 批准（已完成）
→ Terra high Command（已完成任务分解）
→ Luna high Build / Luna medium Inspect（尚未执行）
→ Terra 汇总
→ root 最终验证
```

截至本快照，当前宿主只暴露 `gpt-5.6-sol` 和 `gpt-5.6-terra`，没有暴露
`gpt-5.6-luna/high` 或 `gpt-5.6-luna/medium`。新会话必须重新读取宿主路由证据；若 Luna
仍缺失，停止对应 Build/Inspect 包，不得用 Terra 冒充 Luna，也不得启动 probe child 猜测路由。

---

## 1. 两项关联任务

### 1.1 已执行的 RIA postprocess 检查

用户执行过：

```bash
python3 scripts/run_pipeline.py \
    --config configs/hecheng_ria_0805.yaml \
    --python /home/user/miniconda3/envs/mfa-dev/bin/python \
    --step postprocess --overwrite
```

该任务的重点不是重新运行 ASR/MFA，而是检查 postprocess 输出中的 passed 结果是否存在错误，
尤其是：

- `0 pinyin tokens vs N reference CJK chars`；
- NVV 被误判、删除、重复或污染其他 tier；
- 参考标点丢失、增加或重排；
- 句首要求添加的 `sp1` 被误删或被错误计入词面；
- passed/filtered 集合缺失、重叠或混入 stale 文件；
- report 没有覆盖全部预期 stem。

RIA 配置当前有 `ctc_prealign.nvv_enabled: false`。这表示参考文本模式关闭 ASR NVV 检测，
并不表示参考文本已有 NVV、参考标点或规则要求的句首 `sp1` 应被删除。

### 1.2 English strict v4 fresh-rerun 任务

目标配置：`configs/hecheng_english_mfa.yaml`。

目标是把 `/mnt/Raw/新版合成英文数据` 适配到最新严格管线，重新生成有完整来源证据的 CTC
对齐，再执行 MFA 和 postprocess。当前固定生产 run root 是：

```text
/mnt/nvme3/mfa_runs/hecheng_english/20260806_strict_v4_0
```

当前已知源集合：

- WAV：54,000；
- TXT：53,998；
- 权威 WAV+TXT stem：53,998；
- 缺参考文本 WAV：2；
- TXT-only：0；
- 本次 strict v4 仅处理 53,998 个权威 stem。

配置中的 ready evidence `sha256` 和 `taxonomy_sha256` 仍是
`REPLACE_AFTER_FINALIZE`，这是有意保留的 fail-closed 门。修复、测试、fresh prepare、CTC rerun、
finalize 和独立 verifier 全部通过前，不得替换这两个值，也不得运行完整 English pipeline。

---

## 2. 已冻结的任务目标与非目标

### 2.1 当前 English 任务的内容权威

本任务是严格的 **reference-only forced alignment**：

1. 参考文本是 CJK、英文、普通 lexical token、NVV 和标点的唯一内容权威。
2. ASR/CTC 只为参考 token 提供声学时间边界。
3. 本任务禁止 ASR 新增 NVV。
4. 参考文本已有 NVV 必须保留，标签、数量和顺序不得改变。
5. 标点来自参考文本；其时间可由强制对齐锚点确定，但不得从自由 ASR 文本增加、删除或重排。
6. 句首 `sp1` 是允许且要求保留/添加的确定性结构标签，不属于 ASR 新增词面。
7. 参考文本和音频不匹配时应硬失败并保留证据，不得回退到 ASR 猜测内容。
8. 其他非本任务模式仍可支持 ASR-added NVV；修复不能全局删除该能力。

### 2.2 passed 与 filtered

当前暂时不评价 filtered 内部的内容质量，但以下集合契约始终是硬门：

```text
passed ∩ filtered = ∅
passed ∪ filtered = expected
每个 expected stem 在 postprocess_report.jsonl 中恰好一条最终记录
```

“忽略 filtered”不等于允许漏 stem、额外 stem、重复 stem、stale 文件、异常退出或 provenance
错误。任何这些全局错误都必须让整批失败。

passed 集合必须在已定义的机器可验证契约内没有已知错误。任何 `0 pinyin`、reference mismatch、
额外 NVV、非法时间轴、坏 TextGrid 或缺失证据的 stem 均不得进入 passed。

---

## 3. `0 pinyin tokens vs N reference CJK chars` 的含义与已知根因

该错误表示参考文本中存在 N 个应映射到拼音词面的 CJK 字符，但最终权威 lexical tier 中没有
任何合法声调拼音 token。合法 NVV、参考标点和句首 `sp1` 必须从 CJK/pinyin 计数中排除，不能
因为这些标签存在就触发此错误。

已确认的一条根因链是：

```text
合法声调拼音 rui4
→ 数字/中文数字转换错误地处理拼音尾部声调数字
→ rui四
→ MFA/词典无法识别，可能退化为 unknown/spn
→ 后处理又将 unknown 当作非 lexical/标点类项目
→ 最终合法拼音 token 数量变成 0
```

修复方向：

- 数字规范化不得处理已识别的声调拼音 token；
- CJK→tone-pinyin 投影与普通数值 ITN 必须有明确顺序或保护占位；
- unknown/spn 不能仅凭“不在词典”就被当作标点；
- strict auditor 必须从参考文本独立重算 CJK lexical projection；
- 触发该错误的 stem 可以进入 filtered，但绝不能进入 passed。

---

## 4. 当前代码中的核心问题

### 4.1 P0：`--no-nvv` 语义不足且未被 English rerun 强制

涉及：

- `scripts/ctc_prealign.py`；
- `scripts/prepare_hecheng_english_ctc_ready.py::render_rerun_command()`；
- `scripts/verify_hecheng_english_ctc_ready_v4.py` 的 `expected_command`。

当前 `--no-nvv` 只令 `enable_nvv=False`，即关闭 blank-frame NVV bias。它没有阻止原始 CTC
logits 自然把 NVV ID 选为 argmax。因此即便带了该 flag，自由 ASR 解码仍可能产生 NVV。

同时，English v4 准备器当前生成的固定 rerun command 没有 `--no-nvv`，独立 verifier 的预期
command 也没有。实际 all-GPU 子进程能够在父参数存在时转发 `args.no_nvv`，但父级生产命令没有
强制它。

要求：

- v4 `render_rerun_command()` 必须包含且只包含一次 `--no-nvv`；
- 独立 verifier 必须要求完全相同的 argv；
- 每个 all-GPU child 必须继承该 flag；
- 缺失、重复、语义漂移或任一 shard 未继承均失败；
- run receipt 记录 `nvv_mode=reference_only`、`asr_nvv_bias=false`、
  `content_authority=reference`。

### 4.2 P0：自由 ASR 文本污染 required sidecar

`scripts/ctc_prealign.py` 当前从 `r["text_asr"]` 写 `_text_raw.txt`，再从这份文本转换
`_text_cn.txt`。这允许 ASR-only NVV、CJK、英文或标点进入必需六件套。

必须采用三层防护：

1. **Decoder**：`--no-nvv` 时，仅在自由解码使用的 logits clone 上把 NVV ID 区间设为
   `-inf`。forced alignment 继续使用干净原 logits 和 reference target，因此参考已有 NVV 仍能
   对齐。
2. **Writer**：required artifact 不得从自由 `text_asr` 获取词面。自由解码只能丢弃或放在
   ready namespace 之外的 diagnostic/log。
3. **Verifier**：从 `_ref.txt` 独立重算允许的 lexical/NVV/标点序列，而不是只相信 argv flag
   或生成器自报字段。

### 4.3 七类文件的唯一来源契约

| 文件 | 允许来源 |
| --- | --- |
| `_ref.txt` | 权威 TXT 的 canonical bytes，建议 `strip()+"\n"`；普通 copy、hash 相同、不能 inode alias |
| `.lab` | 参考文本的确定性 lexical 投影：tone-pinyin/CJK、English、reference NVV；不包含普通标点 |
| `.TextGrid` words | 与 `.lab`、tokens 完全相同的参考 lexical token 及声学时间轴 |
| `_tokens.jsonl` | 同一参考 lexical token 序列；ASR 只贡献边界 |
| `_punct.json` | 只包含参考标点及其时间；不得采用自由 ASR 标点 |
| `_text_raw.txt` | canonical reference 内容，不采用自由 ASR decode |
| `_text_cn.txt` | canonical reference 或有版本、可被独立重算的确定性 reference transform |

硬不变量：

```text
.lab tokens == _tokens.jsonl.word == TextGrid 非空 words
reference NVV 原位保留且不重复
reference punctuation 标签顺序完全相同
句首 sp1 满足固定 grammar
_text_raw/_text_cn 不包含任何 reference 外内容
```

### 4.4 Case 97：最终 manifest 在归一化前冻结

英语、`ria`、数字或标点 normalizer 可能改写 `.lab`、tokens 和 TextGrid 的词面、数量与边界。
旧逻辑先生成 manifest，再执行这些 normalizer，导致 manifest `_words`/`n_words` 与最终 bundle
不一致。all-GPU 只会合并这些 stale shard manifest。

要求：所有 normalizer 成功后，从最终 tokens 并与最终 TextGrid/WAV 交叉检查，重新构建
manifest；最终 bundle 全量验证成功后才能原子发布 manifest 和 marker。

### 4.5 Case 98：encoder 60ms 网格冒充物理 WAV 轴

模型帧数推导的 `(frames - query) * 60ms` 只是近似值，不是 WAV header 的
`nframes / sample_rate`。真实抽样中大量 WAV 不落在 60ms 网格，例如 9.44s。

要求：

- 物理 WAV duration 是 TextGrid、tier、token/pause/punct endpoint 和 manifest 的唯一全局轴；
- 非有限、负值、start 越域、非正时长直接失败；
- 仅允许不超过一个 encoder frame加 epsilon 的末端量化误差裁到 WAV 末端；
- 大于阈值的越界失败，不能静默 clip；
- 裁剪后必须重算 duration/label，且不能制造零长区间。

### 4.6 Case 99：模型路径已固定，但模型文件树未绑定

准备器目前只冻结 model path 字符串，没有证明运行时该目录包含哪一版模型；同一路径下替换
`model.pt` 仍可能通过现有证据。

要求：

- prepare 记录模型目录 exact regular-file tree：相对路径、size、SHA-256 和稳定 tree digest；
- 拒绝 symlink、目录逃逸、extra/non-regular/重复项；
- 实际 CTC 进程在完整成功后原子写 run receipt；
- receipt 绑定实际 argv、Python、模型 tree digest、词典 digest、输入/成功 stem 集及 digest；
- all-GPU 所有 shard 必须使用相同模型和词典身份；
- finalize 与独立 verifier 重算并交叉核对 prepare、receipt 和当前文件树。

### 4.7 Case 100：blank-run pause 坐标包含 query frames

模型含 4 个 query frames，每帧 60ms。旧 blank-run 在完整 `raw_y` 上记录坐标，却直接换算成
秒，造成 pause 整体向后偏移约 240ms，尾部还可能越过 WAV。

要求：blank-run 必须先与 speech slice 相交并减去 query frames，再转为秒；跨 query/speech
边界的伪 run 不得保留。之后统一走 Case 98 的 WAV-axis endpoint gate。

### 4.8 Case 101：all-GPU 合并不是严格事务

当前风险包括：

- 主目标同名文件存在时静默 skip；
- shard manifest 解析失败只 warning 后继续；
- 未在移动文件前证明 shard stem 集互斥且并集等于 expected；
- shard `.ctc_normalized` 可能过早复制到父目录；
- 父级在最终全量验证前写 manifest/summary；
- 中途失败可能留下混合来源文件和看似完成的 marker。

要求：先在 staging/内存完整验证每个 shard 的 namespace、manifest、stem 集、普通文件类型和
artifact 一一对应，再证明 shard 集互斥且并集精确，最后事务提交。主目标发生任何碰撞必须失败。
父 marker 必须是最后一个原子发布项；失败不得留下父 final marker/manifest。

### 4.9 Cases 76/83：MFA 输出验证与进程异常仍未闭环

只搜索 `name = words` / `name = phones` 字符串不足以证明 TextGrid 合法。截断文件、重复 tier、
缺 phones、NaN、倒置 interval 或越过音频范围仍可能包含这些字符串。

要求单进程与 sharded MFA 共用同一个严格 parser/validator：

- 唯一且必要的 words/phones tiers；
- 完整 long TextGrid grammar；
- 所有时间有限、正时长、单调、域内；
- exact expected/produced stem set；
- Popen OSError、超时、信号退出、非零返回统一形成结构化失败；
- 每 shard 持久日志和 stem manifest；失败现场不清理；
- 任一失败禁止进入 postprocess。

### 4.10 v3 诊断路径与 v4 生产路径共存

`prepare_hecheng_english_ctc_ready.py` 前半保留 v3 diagnostic 实现，后半定义独立 v4 production
入口，文件末尾调用的是 v4 `main()`。修改时必须确认目标函数属于 v4，不能因为旧 v3 测试或旧
helper 绿色，就判断 v4 契约已经满足。

现有 `verify_prepare_hecheng_english_ctc_ready.py` 中仍包含旧 v3 断言，例如旧 fixture 期望
rendered command 不含 `--no-nvv`。实现 v4 reference-only 后，需要区分/更新测试，防止旧断言
与新生产契约互相掩盖。

---

## 5. 当前工作树和验证状态

当前工作树有大量既存未提交修改，至少涉及：

```text
REGRESSION_ARCHIVE.md
configs/hecheng_english_mfa.yaml
configs/hecheng_ria_0805.yaml
scripts/adjust_ctc_boundaries.py
scripts/align_english_mfa.py
scripts/ctc_prealign.py
scripts/normalize_english_tokens.py
scripts/pad_silence_edges.py
scripts/pipeline_utils.py
scripts/postprocess_textgrids.py
scripts/run_pipeline.py
```

不得执行 `git reset --hard`、`git checkout --` 或覆盖这些改动。开始 Build 前先对当前 diff 做只读
盘点，只修改任务拥有的行，并在最终 handoff 中列出 scoped diff。

中断的早期实现曾部分触及：物理 WAV duration、query-frame pause、final manifest rebuild 等。
这些修改没有完成全套验收，不能标记为已修复。已知仍开放的分支包括 pause 静默 clamp/drop、
punct 末端越界、VAD token end 未统一校验、all-GPU 非事务合并以及不完整 receipt。

历史上以下两组测试曾绿色：

```text
python scripts/verify_prepare_hecheng_english_ctc_ready.py
python scripts/verify_strict_ctc_ready_import.py --verbose
```

当时分别约 17 项和 12 项，但它们没有覆盖新的 reference-only NVV、Cases 97–101 和完整 MFA
parser 故障注入，因此绿色不等于 GO。

`REGRESSION_ARCHIVE.md` 已记录 Cases 97–101，以及 Cases 76/83。reference-only ASR 内容泄漏应在
任何修复前登记为新 Case 102。MFA parser 可继续扩展 Cases 76/83；只有出现独立新根因时才另建
Case 103，避免重复问题编号。

---

## 6. 已批准的串行实施包

共享文件包括 `ctc_prealign.py`、manifest/receipt/evidence schema 和 marker 发布逻辑；这些修改不得
并行写。完整顺序如下：

1. **Luna high Build — reference-only CTC**  
   修改 `ctc_prealign.py`、v4 prepare/verifier 和专属测试；实现 mandatory `--no-nvv`、free-decode
   NVV mask、reference-derived artifacts、NVV/punct/sp1 semantic checks。
2. **Luna medium Inspect — reference-only 审计**  
   只读逐 stem 比对 reference、lab、tokens、TextGrid、punct、text sidecars；检查集合守恒。
3. **Luna high Build — Cases 101+97、98+100**  
   完成 all-GPU transaction、marker-last、后归一化 manifest、WAV-axis finalizer、pause/punct/VAD
   endpoint policy。
4. **Luna medium Inspect — 发布与时间轴审计**  
   只读验证没有 stale marker/manifest、namespace 精确、所有 endpoint 域内。
5. **Luna high Build — Case 99 receipt**  
   实现 model tree、实际运行身份、词典、argv、stem 集和 shard 一致性证据。
6. **Luna medium Inspect — receipt/evidence 审计**  
   只读重算 hash、tree 和 exact stem set；验证失败时没有成功 receipt。
7. **Luna high Build — MFA strict parser**  
   闭环 Cases 76/83，统一 single/sharded validator 和结构化进程失败。
8. **Luna medium Inspect — 最终无 GPU 证据检查与运行监控**  
   汇总全套测试；真实 GPU/MFA 只有在 root 解除 NO-GO 后才能启动和监控。

任一包发现需要改变“reference 是内容权威”、schema 公共接口或发布事务规则时，停止 Build，回到
Sol Strategy 和 root 批准，不得自行扩大设计。

---

## 7. 必须增加的故障测试

### 7.1 Reference-only NVV 和内容来源

- reference 无 NVV，mock 原始 logits 自然 argmax 为 NVV：七类文件均不得出现 NVV；
- monkeypatch `text_asr` 含 `[LAUGHTER]`：任何 required artifact 不得变化；
- reference 含一个 `<LAUGHTER>`：所有应含 lexical content 的容器恰好保留一次；
- 向任一 ready sidecar 注入 extra NVV：finalize 和独立 verifier 均非零；
- 删除、重复 `--no-nvv` 或让一个 shard 不继承：失败；
- 非本任务 fixture 不带 `--no-nvv` 时仍支持 ASR-added NVV；
- mock ASR CJK/English/标点与 reference 完全不同：required artifact 仍完全来自 reference；
- 分别删除、添加、重排 reference NVV、标点和 lexical 序列：失败；
- 合法 NVV、标点、句首 `sp1` 不触发 `0 pinyin`。

### 7.2 Cases 97–101

- 强制 English 或 `ria` normalizer 改变词数，最终 manifest 必须等于最终 bundle；
- duplicate shard stem、坏 manifest、缺/多 artifact、目标碰撞、提前 marker、父验证中途失败：
  均非零且没有父 final marker/manifest；
- 1.00s、9.44s 等非 60ms-grid WAV：TG/tier/manifest 精确使用 WAV duration；
- query/speech 边界 blank-run 从 speech 轴 0 开始，不得偏移 240ms；
- 小于等于允许阈值的尾部量化越界、超过阈值、裁后零长、punct near-end、VAD end 越界；
- 同 model path 替换文件、symlink、extra/non-regular、shard tree digest 不同、argv/词典/stem digest
  漂移：全部在 MFA 前失败。

### 7.3 MFA 和最终集合

- 截断 TextGrid、重复 words、缺 phones、NaN、倒置、越过 WAV；
- single 和 sharded 两条路径都必须失败；
- 预期 18,000、只输出 17,999 必须失败；
- passed/filtered 缺失、额外、交集、stale 文件或 report 重复/漏行必须失败。

---

## 8. 从当前状态到实际运行的完整执行顺序

以下步骤严格串行。标注“修复后”的命令在对应代码和 fault tests 完成前不得运行生产目录。

### Gate 0：重新确认边界和路由

1. 阅读本文，不重新设计 reference-only 目标。
2. 只读检查当前 git diff，保护已有修改。
3. 检查宿主是否实际暴露 Sol/Terra/Luna exact model/effort pair。
4. Luna 缺失时停止 Build/Inspect，不允许替代路由。
5. 确认没有正在运行或复用旧 root 的 CTC/MFA 进程。

### Gate 1：先补回归档案

在 `REGRESSION_ARCHIVE.md` 追加 Case 102，完整记录：

- `--no-nvv` 只关闭 bias；
- 自然 ASR NVV 仍可能产生；
- `_text_raw/_text_cn` 接收自由 ASR 内容；
- reference-only 三层防护；
- fault tests 和停止条件。

只记录问题不表示已经修复。

### Gate 2：完成串行 Build/Inspect 包

按第 6 节顺序实施，每个 Build 后必须先完成对应只读 Inspect。任何 hard test 非零立即停止，不进入
下一包。Terra 汇总全部 handoff 后，由 root 检查最终 integrated diff。

### Gate 3：无 GPU 回归

至少执行：

```bash
git diff --check
python -m compileall -q scripts check_ipa_mapping.py verify_risks.py
python scripts/verify_prepare_hecheng_english_ctc_ready.py
python scripts/verify_strict_ctc_ready_import.py --verbose
```

并执行第 7 节新增的 reference-only、transaction、WAV-axis、receipt 和 MFA parser 专项测试。
只有所有返回码为 0、日志无未解释 warning、测试确实覆盖新分支时，才允许进入 fresh canary。

### Gate 4：只读 inventory 检查

```bash
python scripts/prepare_hecheng_english_ctc_ready.py inspect \
    --source-dir /mnt/Raw/新版合成英文数据 \
    --require-expected-counts \
    --expected-wavs 54000 \
    --expected-txts 53998 \
    --expected-authoritative 53998 \
    --expected-missing-refs 2 \
    --expected-txt-only 0
```

输出必须精确匹配预期，并确认两个 missing-reference stem 与冻结清单一致。

### Gate 5：fresh prepare（修复后）

必须使用一个不存在的新 run root。现有固定路径若已经存在或留下失败现场，不得 overwrite/续跑；应换
新的版本化 root，并同步更新配置。示例目标仍以当前配置路径表示：

```bash
python scripts/prepare_hecheng_english_ctc_ready.py prepare \
    --run-root /mnt/nvme3/mfa_runs/hecheng_english/20260806_strict_v4_0 \
    --source-dir /mnt/Raw/新版合成英文数据 \
    --dictionary-source /mnt/local_E/MFA_Pause/repo/dict/mfa_ipa.dict \
    --asr-python /home/user/miniconda3/envs/asr/bin/python \
    --asr-model /mnt/local_E/nvvasr_standalone/models/Multilingual-NVASR \
    --require-expected-counts
```

prepare 必须只创建 run-local audio/reference/dict view、prepare manifest 和冻结证据，不启动 GPU。

### Gate 6：执行 manifest 中的 exact rerun command（修复后）

不得手工删除或增加参数。修复后的命令语义必须等价于：

```bash
/home/user/miniconda3/envs/asr/bin/python scripts/ctc_prealign.py \
    --data-dir /mnt/nvme3/mfa_runs/hecheng_english/20260806_strict_v4_0/reference_view \
    --audio-dir /mnt/nvme3/mfa_runs/hecheng_english/20260806_strict_v4_0/audio_view \
    --pinyin-dir /mnt/nvme3/mfa_runs/hecheng_english/20260806_strict_v4_0/reference_view \
    --output-dir /mnt/nvme3/mfa_runs/hecheng_english/20260806_strict_v4_0/ctc_rerun_output \
    --model-path /mnt/local_E/nvvasr_standalone/models/Multilingual-NVASR \
    --dict-path /mnt/nvme3/mfa_runs/hecheng_english/20260806_strict_v4_0/dict/mfa_ipa.dict \
    --all-gpus \
    --no-nvv \
    --no-dict-update \
    --require-fresh-output
```

禁止 `--overwrite`。监控每个 shard 的 PID、退出码、日志、stem manifest、GPU 使用情况和产物增长。
任何 shard 失败立即停止父发布；保留失败 root，不进行手工拼接。

### Gate 7：finalize 和独立 verifier（修复后）

```bash
python scripts/prepare_hecheng_english_ctc_ready.py finalize \
    --run-root /mnt/nvme3/mfa_runs/hecheng_english/20260806_strict_v4_0 \
    --source-dir /mnt/Raw/新版合成英文数据 \
    --dictionary-source /mnt/local_E/MFA_Pause/repo/dict/mfa_ipa.dict \
    --asr-python /home/user/miniconda3/envs/asr/bin/python \
    --asr-model /mnt/local_E/nvvasr_standalone/models/Multilingual-NVASR \
    --require-expected-counts

python scripts/verify_hecheng_english_ctc_ready_v4.py \
    --run-root /mnt/nvme3/mfa_runs/hecheng_english/20260806_strict_v4_0 \
    --source-dir /mnt/Raw/新版合成英文数据 \
    --dictionary-source /mnt/local_E/MFA_Pause/repo/dict/mfa_ipa.dict \
    --asr-python /home/user/miniconda3/envs/asr/bin/python \
    --asr-model /mnt/local_E/nvvasr_standalone/models/Multilingual-NVASR
```

finalize 前必须验证 rerun namespace 精确；finalize 失败不得生成 ready evidence。独立 verifier 必须
不导入 prepare helper，并从磁盘重新计算 semantic、hash、model tree、receipt 和 stem set。

### Gate 8：更新 English 配置证据

只有 Gate 7 成功后，才把 `ctc_ready_evidence.json` 的真实 SHA-256 和 taxonomy SHA-256 写入
`configs/hecheng_english_mfa.yaml` 的 `expected_ready_evidence`。同时核对 run root、audio_view、
reference_view、ctc_ready、dict 和 workspace 都属于同一版本。

配置更新后重新运行独立 verifier；任何 hash 漂移都必须重新失败，不能手工放宽。

### Gate 9：fresh pipeline canary

先用冻结的 canary stems file 和全新 workspace 运行完整 strict stage route。strict v4 不允许只跑
单独 step、不允许 padding、不允许 `--force`/`--overwrite`、不允许输出 staging 发布：

```bash
python3 scripts/run_pipeline.py \
    --config configs/hecheng_english_mfa.yaml \
    --python /home/user/miniconda3/envs/mfa-dev/bin/python \
    --ctc-ready-stems-file /ABSOLUTE/PATH/TO/FROZEN_CANARY_STEMS.txt \
    --no-output-staging \
    --no-cache
```

canary 必须包含：纯英文、混合 CJK/English、reference NVV、标点、句首 `sp1`、`ria`、数字、非
60ms-grid WAV，以及历史 `0 pinyin` 风险样本。

### Gate 10：检查 canary 输出

必须验证：

- canary expected stem 精确守恒；
- passed 中无 `0 pinyin`、reference mismatch、extra NVV、非法时间轴或坏 tier；
- filtered 可以存在，但与 passed 不重叠且并集等于 expected；
- report 每 stem 恰一行；
- CTC ready/evidence 在 pipeline 结束时重新验证且未被修改；
- single/sharded MFA parser 结果一致；
- 无发布目录，因为 canary 使用 `--no-output-staging`。

### Gate 11：fresh full run

只有 canary 所有硬门通过，Sol/root 重新给出 GO，且使用新的空 workspace，才执行不带 canary
stems file 的全量命令：

```bash
python3 scripts/run_pipeline.py \
    --config configs/hecheng_english_mfa.yaml \
    --python /home/user/miniconda3/envs/mfa-dev/bin/python \
    --no-output-staging \
    --no-cache
```

全量运行仍不使用 `--force` 或 `--overwrite`。运行期间监控进程、磁盘、每阶段 expected/produced
集合、MFA shard 日志和结构化失败报告。任一硬门失败立即停止后续 stage。

### Gate 12：最终验收和发布

全量成功后从 integrated root 状态重新执行：

- 独立 ready verifier；
- exact passed/filtered/report stem conservation；
- passed semantic audit；
- TextGrid grammar/tier/time-domain audit；
- manifest/receipt/model/dictionary/source hash audit；
- scoped diff 和日志归档。

本任务要求 `--no-output-staging`，因此不会自动发布到
`/mnt/Raw/新版合成英文数据对齐`。只有用户明确授权版本化发布后，才能执行独立发布步骤；不得覆盖
旧结果目录。

---

## 9. GO / NO-GO 判定

### 保持 NO-GO 的任一条件

- Luna 路由是必需的但宿主未暴露；
- Case 102 尚未登记；
- mandatory `--no-nvv`、decoder mask、reference-derived writer 或独立 semantic verifier 未完成；
- Cases 97–101 任一未闭环；
- Cases 76/83 strict MFA parser 未闭环；
- 新 fault tests 未覆盖或非零；
- production/config evidence 仍为 placeholder；
- run root/workspace 非 fresh；
- 发现 stale、missing、extra、duplicate、overlap、坏 receipt、坏 marker 或 provenance 漂移。

### 可以进入 fresh canary 的最低条件

- 所有已批准 Build/Inspect 包完成；
- root 在 integrated worktree 上重新运行全部无 GPU 测试并通过；
- prepare/finalize/independent verifier 三方契约一致；
- config evidence 使用真实最终 hash；
- Sol/root 给出明确 GO；
- canary root 和 workspace 均不存在；
- 实际执行和监控权限已确认。

### 可以接受 full run 的最低条件

除 canary GO 条件外，还必须满足：

```text
passed ∩ filtered = ∅
passed ∪ filtered = 53,998 个 authoritative expected stems
report 每 stem 恰一行
passed 在 semantic、schema、namespace、time axis、TextGrid、MFA parser、receipt、provenance
等所有机器可验证规则下无错误
```

filtered 的内容质量仍不在当前验收范围，但其集合身份、文件合法性和报告记录必须正确。

---

## 10. 关键文件索引

- `configs/hecheng_ria_0805.yaml`：RIA postprocess/参考文本模式配置；
- `configs/hecheng_english_mfa.yaml`：English strict v4 目标配置；
- `scripts/ctc_prealign.py`：CTC forced alignment、NVV、pause、TextGrid、all-GPU 合并；
- `scripts/prepare_hecheng_english_ctc_ready.py`：v3 diagnostic + v4 prepare/finalize 生产入口；
- `scripts/verify_hecheng_english_ctc_ready_v4.py`：独立 ready verifier；
- `scripts/pipeline_utils.py`：CTC bundle 和公共验证；
- `scripts/run_pipeline.py`：strict ready import、MFA、stage denominator 和发布门；
- `scripts/align_english_mfa.py`：English MFA 及相关 shard 行为；
- `scripts/postprocess_textgrids.py`：postprocess、pinyin/CJK/NVV/标点及 passed/filtered 决策；
- `scripts/audit_strict_ok.py`：最终 strict passed 审计；
- `scripts/verify_prepare_hecheng_english_ctc_ready.py`：准备器回归，含旧 v3 测试；
- `scripts/verify_strict_ctc_ready_import.py`：strict import 无 GPU fixture；
- `REGRESSION_ARCHIVE.md`：历史问题和 Cases 76/83、97–101；修复前追加 Case 102。

本文是当前任务的批准基线。除非用户改变“本 English 任务不允许 ASR 新增 NVV、ASR 只提供参考
token 时间轴”的要求，否则新对话应从未完成 Gate 开始，不应重新打开已经冻结的产品决策。
