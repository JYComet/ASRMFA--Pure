# 当前对齐回归、既有代码修改与后续实施交接

## 元数据

| 字段 | 值 |
|---|---|
| 仓库 | `/mnt/local_E/MFA_Pause/repo` |
| 快照时间 | `2026-08-24T10:43:15Z` |
| Git revision | `4ae8427f4c27ae888299afd44de683476b760eb5` |
| 分支 | `main` |
| 工作树 | 脏；包含大量既有修改、删除和未跟踪文件，不能仅凭 revision 重建当前状态 |
| 仓库指令 | 未发现 `AGENTS.md`；`CLAUDE.md:1-48` 要求逻辑冲突类修复写入 `REGRESSION_ARCHIVE.md` |
| 当前阶段 | 已完成调查与方案收敛；最新发现的生产路径根因尚未实施修复，也尚未完成 fresh 1000 条全链路验证 |

本文是新会话的唯一入口，记录任务目标、用户已经明确的业务规则、此前代码修改、当前真实结果、最新根因、仍需实施的工作及验收方法。行号对应上述脏工作树快照；实施前应按符号重新定位，不得因此覆盖既有修改。

## 1. 任务目标与不可变规则

目标是在固定的 1000 个同源样本上建立可追溯、可重复、失败关闭的 CTC/MFA 对齐链路，并根治英文、静音、标点、owner、派生 tier 和边界显示不一致问题。

用户已经明确的规则如下：

1. CTC 分为两个版本：raw CTC 是不可修改的参考证据；processed CTC 可自由后处理，最终 TextGrid 展示的是调整后的 processed CTC 边界。
2. CTC 通常作为边界基点。MFA 偏差过大时，不能让 MFA 把整个词拖走；应以 CTC 锚点和上下文补偿词边界，再按 MFA 音素比例填充词内音素。
3. MFA 未知词或偏差大不等于无法对齐，但必须保留来源、身份和失败证据，不能伪造 owner。
4. 所有词间 `<sp1>` 都向前合并到前词；`<sp0>` 未知归属时也向前合并；首部静音保留；`<sp2>` 不合并并过滤。
5. 尾部静音若有原文本标点证据，应归入最后标点；不能凭空增加标点。
6. 原文本存在标点且停顿证据被后续步骤顶掉时，应恢复标点；没有静音时不因裁剪前后词而补标点。
7. 标点消失本身不触发过滤；只有标点错位或异常增加才过滤。标点较宽可以合理，不能仅因宽度过滤。
8. phone 重叠时应根据 canonical 词身份、MFA 音素顺序和边界证据判断 owner，不能只做数值裁剪或直接丢音素。
9. 最终 words 边界、点击后显示的 phone/派生 tier 边界和实际 owner 必须一致；需增加筛选，阻止视觉边界与点击边界不统一的结果发布。
10. 参考文本模式取得文本后也必须做数字归一化，再转拼音，例如 `target1` 应归一为 `target一`，不能让数字 `1` 成为英文 provenance 单元。
11. 不能通过修改阈值、删除过滤器或只改几个数值提高通过率；修复必须发生在生产逻辑和证据合同层。
12. 最终结论必须来自 fresh workspace 的完整无跳步 1000 条运行，而不是复用旧上游产物后只跑后处理和 strict audit。

成功标准不是强行达到 1000/1000，而是在不降低质量门槛的前提下消除代码引入的系统性回归，并使每一条通过或过滤都能由当前运行的证据解释。

## 2. 用户最初报告的代表现象

这些样本和现象是回归测试的重要来源：

- `12871`：`kpop` 被拆成两个词。
- `52697`：最后一个“欢”延长到末尾，视觉边界与点击显示不一致，标点消失。
- `52868`：`app` 只剩一个 `p`。
- `53198`：`sos` 第一个 `s` 的音素消失。
- `16806`：词中出现内部 `sp`；用户预期若原本有标点则应恢复并合并，否则异常静音应被处理或过滤。
- `16735`：“设备”的“备”和“就绪”的“绪”出现 words 边界与点击后的 phone/派生 tier 边界不一致。
- 含 `ok` 的音频：怀疑英文音素未逐个正确对齐，部分 `ok` 被偏移到其他词。
- `target1`：数字 `1` 未在参考文本模式中归一为中文数字，污染后续英文 canonical/provenance。

这些问题不是彼此独立的 UI 瑕疵。它们共同指向四类证据级联：词身份被拆分或错配、raw/processed 时间轴混用、后处理结果被后续 tier 重建覆盖、静音/标点/phone owner 的派生规则没有共享同一最终边界。

## 3. 此前已做的代码修改

以下是当前脏工作树中已经存在、需要保留并继续验证的主要改动。它们不能被描述为全部端到端验证完成。

| 改动 | 主要位置 | 当前判断 |
|---|---|---|
| 参考文本数字归一化，确保 reference authority 载入后、转拼音前把阿拉伯数字转为中文数字 | `scripts/pipeline_utils.py:1555-1607` 及 pipeline normalization 调用链 | 方向正确，解决 `target1` 类 identity 污染 |
| 建立 canonical English units、稳定 schema、surface text、source ordinals 和 immutable canonical span | `scripts/english_units.py:1-163` | 方向正确，是英文 provenance 的基础 |
| English MFA ledger 绑定 canonical unit，并保存 canonical/processed 双轴证据 | `scripts/align_english_mfa.py:798-875`, `scripts/align_english_mfa.py:1649-1668` | 方向正确，但 processed geometry 当前会把词错误截短 |
| raw CTC manifest 与 work/processed receipt 分离，禁止后处理覆盖 raw evidence | `scripts/pipeline_utils.py:697-925`, `scripts/run_pipeline.py:1235-1346` | 必须保留 |
| 新增 torch-free processed geometry resolver 和向量化 VAD 搜索 | `scripts/ctc_processed_geometry.py:1-189`, `scripts/run_pipeline.py:2163-2197` | 架构与性能方向正确，resolver 约束仍有缺口 |
| `ctc_prealign` raw producer 不再写 processed 几何，raw TextGrid 保留 canonical span | `scripts/ctc_prealign.py` | raw/processed 分层方向正确 |
| resample 支持 flat/nested speaker WAV，建立 source audio index 与 transform receipt，overwrite 时避免旧 nested 输出污染 | `scripts/run_pipeline.py:692-708`, `scripts/run_pipeline.py:1017-1095`, `scripts/run_pipeline.py:1353-1407`, `scripts/pipeline_utils.py:2529-2589` | 应保留并纳入 freshness 验证 |
| 在 CTC 前冻结 selection，将原始 55998/56000 规模 accounting 投影到固定 1000 eligible stems | `scripts/run_pipeline.py:749-828`, `scripts/run_pipeline.py:1626-1707` | v7/v9 stems 一致，说明分母合同有效 |
| MFA progressive retry、分区失败后的递进重试与 singleton rescue 状态机 | `scripts/run_pipeline.py:2591-2666`, `scripts/run_pipeline.py:5497-5609`, `tests/test_mfa_retry.py:130-195` | 有聚焦测试，不应被本次修复破坏 |
| postprocess/audit 通过 transform receipt 解析 nested TTS 实际音频路径 | `scripts/audit_strict_ok.py:516-599`, `scripts/run_pipeline.py:1459-1476` | 方向正确 |
| strict audit 区分 source canonical ordinal 与实际 CTC words tier ordinal | `scripts/audit_strict_ok.py:1598-1732` | 防止英文拆分后用错误 tier index 对齐 |
| evidence repair 按 label 和边界重新查找 final pair，而不是使用 tier 重建前的旧 index | `scripts/audit_strict_ok.py:997-1078` | 用于避免派生 tier 级联错位 |
| `sp1` audit 保留 raw text/pinyin 首标记，允许派生 tier 在 0 秒开始 lexical；仅当后面仍有 lexical 时判定为内部非法 `sp1` | `scripts/audit_strict_ok.py:1081-1113` | 方向正确 |
| postprocess 已加入 `sp0`/`sp1` 向前合并、`sp2` 保留并过滤、标点恢复、owner/phone 分配和边界一致性逻辑 | `scripts/postprocess_textgrids.py`, `README.md:487-519` | 规则已实现过，但仍需验证是否在后续 rebuild/publication 被覆盖 |
| child 因强制质量过滤返回非零时，顶层仍刷新完整 partition receipt 和 exact set | `scripts/run_pipeline.py:3103-3146`, `scripts/run_pipeline.py:3369-3474` | accounting 修复有效，但不等于对齐质量有效 |
| 新增多组英文、artifact、boundary、punctuation、owner、SP、fresh workspace、subset denominator 测试 | `tests/` 下多个未跟踪或已修改文件 | 覆盖面扩大，但最新发现显示生产 resolver 未被正确覆盖 |

## 4. 当前工作树保护

当前工作树已有大量用户工作，实施者必须先逐文件查看 diff，再打最小补丁。特别注意：

- `scripts/ctc_processed_geometry.py` 和 `scripts/english_units.py` 是未跟踪文件，但已经成为其他修改的运行依赖，不能清理。
- `tests/run_gpu1000_continuation_regressions.py` 处于删除状态，不得擅自恢复或永久清除。
- `README.md`、`REGRESSION_ARCHIVE.md`、多个配置、词典和核心脚本均已修改。
- 新增测试与分析脚本仍有很多未跟踪文件。
- 禁止使用工作树级 reset、checkout、clean 或覆盖式恢复。

新会话开始时必须运行：

```bash
git rev-parse HEAD
git branch --show-current
git status --short
git diff --stat
```

revision 只用于定位基线，不能代表当前实际代码；后续 receipt freshness 必须基于实际文件内容 hash。

## 5. 当前数据结果与有效性边界

### 5.1 v7 基线

路径：

`/mnt/nvme3/mfa_workspace_1000test_rerun_20260821_v7/strict_ok_runs/20260824T030211Z_160646`

| 层级 | 通过 | 过滤 |
|---|---:|---:|
| postprocess | 804 | 196 |
| strict | 792 | 208 |

v7 strict receipt 的 eligible 为 1000，output 为 792，filtered 为 208，`global_reasons=[]`。v7 可作为 matched-stem 回归基线，但不是绝对正确答案。

### 5.2 v9 最新 downstream diagnostic

路径：

`/mnt/nvme3/mfa_workspace_1000test_rerun_20260821_v9/strict_ok_runs/20260824T083035Z_781331`

| 指标 | 数量 |
|---|---:|
| 固定 eligible stems | 1000 |
| postprocess/strict output | 100 |
| postprocess/strict filtered | 900 |
| `global_reasons` | `[]` |

v7 与 v9 的 expected stems 均为 1000，顺序和内容完全一致。因此通过数下降不是抽样集合变化造成的。

v9 postprocess 重叠原因如下，不能直接相加为 900：

- 899 条包含 `english_provenance_rejected`。
- 470 条包含 `ctc_lexical_sequence_mismatch`，同时伴随 `mfa_unknown_source`。
- 373 条包含 `mid_sp`，同时伴随 `strict_interior_sp`。
- 另有 1 条 `sp3`，1 条 `short_word`。

### 5.3 v9 英文证据

`en_alignment_manifest.json` 统计：

| 指标 | 数量 |
|---|---:|
| English words | 3265 |
| verified | 2156 |
| rejected | 1109 |
| `segment_too_short` | 1107 |
| `source_tg_missing_or_ambiguous` | 2 |

进一步扫描所有 `*_en_phones.json`：

- 1258 个 processed English spans 短于 canonical span。
- 异常分布在 807 个 stems。
- 632 个 processed spans 不超过 40 ms。

代表样本 `000009_Mira_欢迎回来_今天也请按照自己的节奏生活_Mira_v-tuber_和_open-ai_会陪你聊一会儿_如果你想安` 的第二个 `Mira`：

| 字段 | 值 |
|---|---|
| canonical span | `4.65-4.95` |
| processed span | `4.65-4.68` |
| processed duration | 30 ms |
| boundary source | `raw_end_long_pause` |
| rejection | `segment_too_short` |

这证明通过率下降主要受系统性 processed geometry 截短影响，不是 1107 个真实英文单词都只有几十毫秒。

### 5.4 为什么当前 100/1000 不能作为最终结论

v9 最终 strict run 的 route 只有：

```text
postprocess -> strict_ok
```

相关 artifact 的 UTC 时间依次为：raw CTC 07:14、adjusted work receipt 07:38、MFA axis 07:41、English manifest 08:08、最终 strict 08:32。最终运行复用了更早的上游产物，没有重新执行 prealign、resample、adjust、align、align_en 等阶段。

因此当前只能断言：固定 1000 分母的 downstream accounting 自洽，100 条通过、900 条过滤，且没有全局集合丢失。不能断言：当前代码已端到端验证、上游修复已生效、100 是真实质量上限、900 条都是素材问题。

`REGRESSION_ARCHIVE.md` Case 171 中把该结果描述为成功的措辞需要在实施修复时纠正。

## 6. 最新确认的根因链

### 6.1 pauses tier 的空标签 speech partition 被误当作 pause

生产 parser：`scripts/adjust_ctc_boundaries.py:279-303` 的 `_read_pause_intervals`。

`scripts/ctc_prealign.py:655-689` 生成的 pauses tier 是完整时间分区：真实 pause interval 使用时长文本作为 label，speech/complement interval 也存在但 label 为空。

当前 `_read_pause_intervals` 只要处于 pauses tier 且 `xmax > xmin` 就追加 interval，没有解析并要求 `text` 非空。因此大量正常语音区间的起点被伪装成 pause 起点。

在代表样本附近，4.08-4.68 是标注 `600.0ms` 的真实 pause，而 4.68-4.98 是空标签 speech partition。当前 parser 两者都返回，resolver 随后把 4.68 选成第二个 `Mira` 的所谓 `raw_end_long_pause`，把 4.65-4.95 错切成 4.65-4.68。

### 6.2 生产 resolver 允许 hard boundary 落在 canonical 词内部

生产调用位于 `scripts/adjust_ctc_boundaries.py:429-439`，实际调用 `scripts/ctc_processed_geometry.py:55-189` 的 `resolve_processed_english_spans`。

当前 resolver：

- 保留 `>= 0.2` 秒长停顿门槛，这个产品阈值本身不是根因。
- 接受满足 `raw_start <= boundary < next_lexical_start` 的 punctuation/pause hard boundary。
- 没有要求 boundary 不早于 canonical raw end。
- 最后只检查 `processed_end > raw_start`，没有检查 `processed_end >= canonical_end`。

因此只要 parser 提供一个位于词内部的错误边界，resolver 就能合法地把 canonical 词切短。

### 6.3 公共 validator 没有守住 canonical end

`scripts/pipeline_utils.py:1921-1968` 检查了 span 数值、正时长、processed start 不早于 raw start和 start identity，但没有验证：

```text
processed_end >= canonical_end
```

生产错误因此没有在 artifact 合同层立即失败，而是传播到 English MFA ledger，最终表现为 `segment_too_short`、English provenance rejection、词序 mismatch、MFA unknown source 和 owner 冲突的级联。

### 6.4 测试覆盖了重复实现，不是生产实现

仓库当前有两个同名 resolver 定义：

- 生产共享实现：`scripts/ctc_processed_geometry.py:55`
- 重复实现：`scripts/ctc_prealign.py:749`

`tests/test_ctc_english_units.py` 导入并测试的是 `ctc_prealign` 中的重复实现，而 production adjust 调用共享实现。现有 hard-boundary 测试只覆盖 canonical 词尾之后的边界，没有覆盖词内部边界。因此测试通过并不能证明生产路径正确。

### 6.5 为什么问题“修了很久反而变多”

此前多个局部修复本身方向正确，但缺少以下闭环：

1. raw/processed 分轴后新增了生产 resolver，却保留旧重复实现，测试和生产发生分叉。
2. 后处理、strict audit 和 accounting 被重复运行并改进，但上游 adjusted/English artifact 没有随最新代码重建。
3. 过滤器更完整后，把上游几何错误暴露成更多 provenance、mismatch、unknown 和 SP 过滤；数量变多是级联暴露，不代表每个新增过滤条件都错。
4. receipt 能证明 1000 分母守恒，却没有 producer/config/dependency fingerprints，不能证明复用 artifact 来自当前代码。
5. 最终只跑 `postprocess -> strict_ok` 跳过了用户要求验证的关键阶段，因此不能用该结果评价前面代码修改。

## 7. Facts、Assumptions、Decisions、Open Questions

### Facts

1. v7 与 v9 使用完全相同的 1000 stems。
2. v7 strict 为 792/208，v9 downstream diagnostic 为 100/900。
3. v9 有 1258 个 processed English spans 短于 canonical，影响 807 stems。
4. `_read_pause_intervals` 当前不要求非空 label。
5. 生产 resolver 当前允许 canonical 内部 hard boundary。
6. 公共 validator 没有 canonical end 下界。
7. 测试使用重复 resolver，而不是 production shared resolver。
8. v9 最终 route 仅为 `postprocess -> strict_ok`。
9. 当前工作树不能由 Git revision 单独重建。

### Assumptions

1. 200 ms 长停顿阈值继续保持，不因本次回归调整。
2. v7 是 matched-stem 回归基线，不是绝对真值。
3. strict provenance、owner partition、标点和 SP 合同不能为提高通过数而放宽。
4. canonical end 之后、next lexical start 之前的真实 hard boundary仍可用于向后延展 processed span。

### Decisions

1. pauses tier 只有去除空白后非空的 label 才代表 pause；空 label partition 必须忽略。
2. 仓库只能保留一个 resolver 算法，生产和测试必须调用同一实现。
3. 成功输出必须满足 `processed_start == canonical_start` 且 `processed_end >= canonical_end`，允许固定的小浮点容差。
4. canonical 内部 hard boundary 应忽略；找不到合法外部边界时保留 canonical end。若 canonical 与 next owner 本身冲突，则结构化失败。
5. raw CTC 不变，所有调整仅写派生 work/processed artifact。
6. 历史 v7/v9 artifact 仅用于审计比较，不能作为 fresh v10 的可恢复上游。
7. 不能关闭 English provenance、MFA unknown、lexical mismatch、SP、标点或 owner 过滤。

### Open Questions

1. v10 最终输出是否使用 `/mnt/Raw/0805test_v10`；运行前需确认目录为空、权限和容量足够。
2. 历史无 fingerprint receipt 的兼容方式；建议只允许审计，不允许认证为当前代码的可恢复产物。
3. GPU、NAS、MFA 完整 1000 条运行窗口何时可用。
4. source corpus 若继续变化，需在 v10 前冻结实际 1000-stem manifest hash，不能只依赖 `pre_ctc_limit: 1000`。

## 8. 实施范围与非范围

### 范围内

- 修复 pause parser 的非空 label 语义。
- 唯一化 processed English geometry resolver。
- 增加 canonical end 下界和结构化 geometry failure。
- 让测试直接覆盖生产调用路径。
- 加强 raw/work/processed artifact validator。
- 增加 receipt producer/config/dependency/input fingerprints 和 stale-resume 拒绝。
- 验证已有数字归一化、英文 canonical identity、SP 合并、标点恢复和 owner 逻辑不会在后续 tier rebuild 中被覆盖。
- 纠正 Case 171 并追加新的回归记录。
- 创建隔离 v10 配置并 fresh 跑完整 1000 条。
- 生成 v7/v9/v10 matched-stem delta。

### 非范围

- 调低 200 ms 阈值。
- 接受 MFA unknown source 或放宽 English provenance。
- 删除 `mid_sp`、strict interior、lexical mismatch、标点或 owner 检查来提升数量。
- 修改或覆盖 v7/v9 历史产物。
- 重置当前脏工作树。
- 在没有 fresh v10 全链路证据时宣告修复完成。

## 9. 编号需求

### R1：显式 pause 语义

`_read_pause_intervals` 只能返回 pauses tier 中 label 去除空白后非空的 interval。parser 负责判定是否为显式 pause，resolver 继续负责判定该 pause 是否达到 200 ms。

### R2：唯一生产 resolver

仓库中只能有一个 `resolve_processed_english_spans` 算法实现。`ctc_prealign`、`adjust` 和测试全部导入 `scripts/ctc_processed_geometry.py` 的共享实现，或通过不含算法的薄适配层调用它。

### R3：canonical 几何下界

每个成功 English unit 必须满足：

```text
processed_start = canonical_start
processed_end >= canonical_end
```

hard boundary 位于 canonical 内部时忽略并继续寻找 canonical end 之后的合法边界；没有外部边界时保留 canonical end。canonical end 已越过 next lexical owner 时返回带 reason code 的结构化失败，不能写短 span、负区间或重叠 owner。

### R4：共享 validator 失败关闭

`scripts/pipeline_utils.py` 必须拒绝 processed end 早于 canonical end、identity 不一致、缺少 raw/processed 双轴证据、geometry failure 无 reason code等情况。错误上下文至少包含 stem、unit id、canonical span、candidate boundary、next lexical start 和 boundary source。

### R5：后处理结果不可被覆盖

对每次 words 边界、SP 合并、标点恢复、phone owner 调整后重建派生 tier 的路径做审查。最终 publication 前重新验证：

- words 与 phone owner 完整覆盖且不重叠。
- 点击看到的 phone/派生 tier 外边界与最终 words 边界一致。
- 已向前合并的 `sp0`/`sp1` 不会由旧 tier 数据重新插回词间。
- `sp2` 保留为过滤证据。
- 无静音时不凭空补标点；有原标点和静音证据时不会被后续 rebuild 顶掉。
- 英文 word 不因 tier rebuild 丢首尾 phone、重复字母或 canonical unit。

### R6：生产路径回归测试

至少覆盖以下场景：

1. pauses tier 同时包含非空真实 pause 和空 speech partition。
2. 199 ms 显式 pause 不成为 long-pause hard boundary，200 ms 成为候选。
3. hard boundary 位于 canonical 内部和 canonical end 之后两种情况。
4. canonical end 与 next lexical start 冲突时结构化失败。
5. `000009` 等价样本不能再产生 `4.65-4.68`。
6. raw span 内容 hash 在 adjust 前后不变。
7. 测试和 production adjust 调用同一 resolver。
8. `ok`、`kpop`、`app`、`sos` 的 canonical 英文单元、字母/音素完整性和 owner。
9. `sp0`、所有词间 `sp1` 向前合并，首静音保留，`sp2` 过滤。
10. 标点消失不单独过滤，标点错位或异常增加才过滤。
11. words/phones/派生 tier 最终边界一致。
12. reference numeral 在转拼音前已归一化，`target1` 不产生孤立英文/数字 provenance。

### R7：receipt freshness

关键 receipt 至少记录 schema version、producer 模块内容 SHA-256、dirty-worktree 标识、effective config SHA-256、Python/MFA/关键依赖版本、模型和词典 hash、reference manifest hash、输入 receipt hash、expected-stems hash、输出 manifest hash与 UTC 时间。

resume 时必须验证整条依赖链。旧 receipt 缺少 fingerprint 时可以只读审计，但不得被 fresh v10 当作当前上游。

### R8：文档证据

更正 `REGRESSION_ARCHIVE.md` Case 171：v9 只是 downstream diagnostic，不是端到端成功。追加 Case 172，记录空 partition、canonical 内部 hard boundary、重复 resolver、validator 缺口、修复前后数据和复现命令。

### R9：隔离 v10 全链路

新增而不覆盖旧配置，例如 `configs/hecheng_en_1000_test_v10_fresh.yaml`：

- workspace 使用 `/mnt/nvme3/mfa_workspace_1000test_fresh_20260824_v10`。
- 使用新的 output 目录。
- `require_fresh_workspace: true`。
- 固定相同 1000 stems。
- 完整启用 adjust、MFA、English MFA、postprocess 和 strict。
- 保持 `min_segment_dur_ms: 200` 及严格过滤合同。

完整 route 至少包含：

```text
pad_silence
prealign
normalize_punct
normalize
normalize_ria
normalize_en
resample
adjust
align
align_en
postprocess
strict_ok
```

不得以 `--skip-to`、只跑 postprocess、旧 checkpoint 或旧上游缓存形成发布结论。

### R10：matched-stem 差异报告

报告必须包含 v7/v9/v10 的 expected-stems hash、逐 stem 状态、postprocess/strict 数量、重叠过滤原因、processed span 短于 canonical 的数量、v7 通过而 v10 失败的清单、v9 失败而 v10 恢复的清单及 v10 新增回归。

## 10. 受影响文件与实施入口

| 文件 | 位置/符号 | 动作 |
|---|---|---|
| `scripts/adjust_ctc_boundaries.py` | `_read_pause_intervals` 约 279-303；resolver 调用约 429-439 | 只采集非空 pause label，传播结构化 geometry 结果 |
| `scripts/ctc_processed_geometry.py` | `resolve_processed_english_spans` 约 55-189 | canonical end 下界、内部 hard boundary 处理、结构化失败 |
| `scripts/ctc_prealign.py` | pauses tier producer 约 655-689；重复 resolver 约 749-898 | 保留完整 partition；删除重复算法或改为共享导入 |
| `scripts/pipeline_utils.py` | raw/work receipt 约 697-925；span validation 约 1921-1968；transform receipt 约 2529-2589 | validator 和 fingerprints |
| `scripts/run_pipeline.py` | resample、derived axis receipt、adjust、postprocess accounting、MFA retry、fresh workspace | 接入 freshness 校验，保证完整 route |
| `scripts/align_english_mfa.py` | canonical ledger 约 798-875；word evidence 约 1649-1668 | 保留双轴，传播 geometry failure |
| `scripts/postprocess_textgrids.py` | SP、标点、owner、边界调整与 tier rebuild | 审查所有后写覆盖点并加 publication invariant |
| `scripts/audit_strict_ok.py` | evidence repair 约 997-1078；SP1 约 1081-1113；ordinal 约 1598-1732 | 保持严格审计并验证最终边界一致性 |
| `tests/test_ctc_english_units.py` | 当前 resolver 导入和 hard-boundary tests | 改为 production shared resolver，新增内部边界与 pause label fixture |
| `tests/test_ctc_artifact_versions.py` | artifact contracts | fingerprints 与 stale resume |
| `tests/test_axis_contracts.py` | raw/processed axis | end 下界和 raw immutability |
| `tests/test_align_english_mfa_canonical_units.py` | English ledger | structured failure 和 canonical 保留 |
| `tests/test_boundary_punctuation_display_regressions.py` | 标点与视觉/点击边界 | 加强 publication invariants |
| `REGRESSION_ARCHIVE.md` | Case 171 及末尾 | 纠正结论并追加 Case 172 |
| `configs/hecheng_en_1000_test.yaml` | 旧基线配置 | 只读参考，不覆盖 |
| 新 v10 配置 | fresh workspace/output | 全链路验证 |

## 11. 有序实施计划

1. 保存当前 status 和每个目标文件 diff；确认两个未跟踪核心模块仍在。
2. 先补会暴露当前故障的生产路径测试：空 pause partition、内部 hard boundary、canonical end、代表 `Mira` 几何。
3. 修复 `_read_pause_intervals`，不改变 200 ms 阈值。
4. 唯一化 resolver，实施 canonical end 下界与结构化失败。
5. 加强共享 validator 和 English ledger failure 传播。
6. 审查 postprocess 的全部 mutation/rebuild/publication 路径，确保 SP、标点、owner、phone 和 words 最终边界不会被旧数据覆盖。
7. 补 `ok`、`kpop`、`app`、`sos`、数字归一化和视觉/点击边界回归测试。
8. 增加 receipt producer/config/dependency fingerprints 与 stale-resume 拒绝。
9. 运行聚焦测试、完整测试、compile 和 diff 检查。
10. 更正 Case 171，追加 Case 172，并保持 README 合同不被弱化。
11. 创建独立 v10 fresh 配置，冻结与 v7/v9 完全相同的 1000-stem hash。
12. 执行完整无跳步运行，验证每个 stage 的 fingerprint 和 route。
13. 扫描 processed/canonical 几何、SP、标点、owner、tier 边界和 English provenance。
14. 生成 v7/v9/v10 逐 stem delta；只有全部验收通过后才更新完成结论。

阶段依赖：

```text
frozen source selection
  -> pad_silence
  -> prealign / immutable raw CTC / labeled pauses partition
  -> normalization
  -> resample / source transform receipt
  -> adjust / shared processed resolver / derived CTC receipt
  -> align / MFA axis receipt
  -> align_en / canonical + processed English ledger
  -> postprocess / SP + punctuation + owner + tier publication
  -> strict_ok / provenance + final accounting
  -> v7/v9/v10 matched-stem delta
```

## 12. 需求与验收追踪

| 需求 | 验收 |
|---|---|
| R1 | AC-01、AC-02 |
| R2 | AC-03 |
| R3 | AC-04、AC-05、AC-06 |
| R4 | AC-05、AC-07 |
| R5 | AC-08、AC-09、AC-10 |
| R6 | AC-01 至 AC-10 |
| R7 | AC-11、AC-12 |
| R8 | AC-13 |
| R9 | AC-14、AC-15 |
| R10 | AC-16、AC-17 |

### AC-01：pause parser

空 label speech/complement interval 不出现在 pause 列表；非空真实 pause 保留。

### AC-02：200 ms

199 ms 显式 pause 不成为 long-pause hard boundary；200 ms 显式 pause 成为候选，阈值没有降低。

### AC-03：单一 resolver

`rg -n '^def resolve_processed_english_spans' scripts tests` 只返回一个算法定义，production 和 tests 使用同一实现。

### AC-04：canonical 几何

所有成功 English evidence 均满足 `processed_end + tolerance >= canonical_end`；v10 全量违规数为 0。

### AC-05：失败关闭

canonical owner 冲突时输出稳定 reason code 和完整上下文，不写短 span、负区间、静默丢词或伪造 owner。

### AC-06：代表样本

`000009` 第二个 `Mira` 不得再出现 canonical `4.65-4.95`、processed `4.65-4.68`、source `raw_end_long_pause` 的组合。成功结果的 end 不早于 4.95，否则必须结构化失败。

### AC-07：raw immutable

adjust 前后 raw CTC manifest 与 canonical spans 内容 hash 不变，处理结果只写 derived/work artifact。

### AC-08：SP、标点和 owner

所有词间 `sp0`/`sp1` 已按规则向前合并，首静音保留，`sp2` 被过滤；标点消失不单独过滤，错位/异常增加被过滤；words、phones 和派生 tier owner 无重叠且完整覆盖。

### AC-09：英文完整性

`ok`、`kpop`、`app`、`sos` 不再因英文拆分或 tier rebuild 丢 canonical unit、重复字母对应 phone 或首尾 phone。异常 MFA 可按 CTC 锚点回填，但 provenance 必须完整。

### AC-10：显示一致性

最终 TextGrid 中 words 外边界与点击后的 phone/派生 tier owner 边界一致；检测器能过滤任何后写覆盖导致的不一致。

### AC-11：fingerprints

关键 v10 receipts 包含 producer、effective config、dependencies、inputs、expected stems 和 outputs 的 fingerprints。

### AC-12：stale resume

修改 producer、配置、词典、模型或上游 receipt 后，resume 明确失败并指出不匹配字段。

### AC-13：文档

Case 171 不再声称 v9 端到端成功；Case 172 包含现象、根因链、函数位置、修复前后数值和复现命令。

### AC-14：fresh route

v10 workspace/output 与 v7/v9 隔离，receipt 列出完整 route，每个 stage fingerprint 属于同一条 v10 依赖链。

### AC-15：1000-stem accounting

v10 `eligible_count = 1000`，`output_count + filtered_count = 1000`，`global_reasons = []`，expected stems 与 v7/v9 顺序和内容一致。

### AC-16：系统性回归消失

v10 中 processed span 短于 canonical 为 0、空标签 pause 导致的 `segment_too_short` 为 0、canonical 内部 `raw_end_long_pause` 截短为 0。真实短段仍可严格失败，但必须有独立有效证据。

### AC-17：相对基线发布门槛

若 v10 strict accepted 少于 v7 的 792，或 v7 accepted stem 在 v10 因 English provenance、CTC/MFA mismatch、内部 SP 或 canonical shrinkage 被拒绝，则不能宣告完成；必须逐 stem 解释，不能降低过滤条件换数量。

## 13. 验证命令

从仓库根目录执行。

### 单一 resolver 与静态检查

```bash
rg -n '^def resolve_processed_english_spans' scripts tests
rg -n 'resolve_processed_english_spans' scripts/adjust_ctc_boundaries.py scripts/ctc_prealign.py tests
python -m compileall -q scripts
git diff --check
```

预期：只有 shared 模块定义 resolver；所有消费者使用同一实现；编译和 whitespace 检查通过。

### 聚焦测试

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_ctc_english_units.py \
  tests/test_ctc_artifact_versions.py \
  tests/test_axis_contracts.py \
  tests/test_align_english_mfa_canonical_units.py \
  tests/test_boundary_punctuation_display_regressions.py \
  tests/test_postprocess_geometry.py \
  tests/test_postprocess_recovery_geometry.py \
  tests/test_postprocess_word_energy.py \
  tests/test_run_pipeline_subset_denominator.py \
  tests/test_mfa_retry.py \
  tests/test_audit_sp3.py
```

预期：全部通过；新增测试直接命中 production shared resolver，并覆盖 pause label、canonical 内部边界、SP、标点、owner、英文完整性和显示一致性。

### 完整测试

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```

预期：零失败，不产生仓库内 pytest cache，不破坏 strict provenance、MFA retry、nested audio axis、partition 或 denominator 合同。

### v10 完整运行

```bash
PYTHONDONTWRITEBYTECODE=1 \
/home/user/miniconda3/envs/mfa-dev/bin/python \
scripts/run_pipeline.py \
  --config configs/hecheng_en_1000_test_v10_fresh.yaml \
  --python /home/user/miniconda3/envs/mfa-dev/bin/python
```

预期：不使用 `--skip-to` 或只跑下游的 step；route 包含完整阶段；1000 分母守恒；所有 artifacts 的 fingerprints 构成闭合的新依赖链。

### v10 几何扫描

```bash
V10_WORK=/mnt/nvme3/mfa_workspace_1000test_fresh_20260824_v10
find "$V10_WORK/en_phones" -maxdepth 1 -type f -name '*_en_phones.json' \
  -exec jq -r '
    .stem as $stem
    | .segments[].words[]
    | select((.end + 0.000001) < .canonical_span[1])
    | [$stem, .start, .end, .canonical_span[0], .canonical_span[1], .source]
    | @tsv
  ' {} +
```

预期：无输出。

### 最终检查

```bash
git diff --check
git diff --stat
git status --short
```

预期：只有计划内目标文件产生新增差异，所有任务开始前的用户改动、未跟踪文件和删除状态均被保留。

## 14. 风险、回滚与谨慎事项

| 风险 | 影响 | 缓解 |
|---|---|---|
| 删除重复 resolver 时遗漏调用方 | 生产与测试再次分叉 | 用 `rg` 验证单一定义和全部调用 |
| pause parser 过严，旧格式 label 未识别 | 合法延展减少 | 用实际 pauses tier fixture 覆盖；未知格式结构化报告 |
| canonical end 与 next owner 冲突 | 词重叠 | fail closed，不强行钳制跨 owner |
| postprocess 修复再次被 tier rebuild 覆盖 | 视觉/点击边界不一致 | 每次 rebuild 后执行 publication invariant，最终写盘前再审计 |
| fingerprints 使历史 resume 失效 | 旧 workspace 不能继续认证 | 允许历史只读审计，v10 使用 fresh workspace |
| v10 通过率仍低 | 可能还有真实数据问题或其他代码回归 | 用逐 stem delta 分析，不关闭过滤器 |
| 无关 dirty 文件被覆盖 | 用户工作丢失 | 最小 patch、逐文件 diff、禁止 reset/checkout/clean |

回滚只能针对本次新增 patch 使用审阅后的反向补丁，不能对整个工作树 reset。v7/v9 artifacts 保持只读。v10 运行失败时先保留 receipt 和失败证据；删除 v10 目录属于破坏性操作，需要单独确认。

重要不变量：

1. raw CTC 永不被 adjust、MFA 或 postprocess 覆盖。
2. canonical span 是词身份和最小几何范围，不是可被 VAD/pause 任意缩短的建议值。
3. processed span 可向合法停顿延展，但不能向 canonical 内部收缩。
4. 空标签 interval 是 partition 补集，不是 pause。
5. 200 ms 阈值保持不变。
6. `global_reasons=[]` 只表示全局 accounting 没有附加错误，不表示每个 stem 质量通过。
7. 1000 是 frozen denominator，不是强制 accepted 数量。
8. mtime 不能替代 producer/config/dependency fingerprint。
9. 历史 artifact 可比较，但不能混入 fresh v10。

## 15. 当前阻塞项与 readiness

| Gate | 状态 | 说明 |
|---|---|---|
| 历史事实与 matched stems | 通过 | v7/v9 的集合、数量和主要原因已核验 |
| 最新根因定位 | 通过 | pause parser、resolver、validator、重复测试路径形成闭环 |
| 根因代码修复 | 未通过 | 尚未实施 R1-R4 |
| 后处理覆盖链复审 | 未通过 | 需要按 R5 全面检查 mutation/rebuild/publication |
| 生产路径回归测试 | 未通过 | 当前测试仍覆盖重复 resolver |
| receipt freshness | 未通过 | 完整 fingerprints 尚未实现 |
| v10 fresh 配置 | 未通过 | 尚未创建 |
| v10 full route | 未通过 | 尚未运行 |
| v7/v9/v10 delta | 未通过 | 缺少 v10 候选 |
| 发布 readiness | 未通过 | 必须满足 AC-01 至 AC-17 |

当前决策：可以进入实施窗口，但不能宣告修复完成，也不能把 v9 的 100/1000 描述为成功。

## 16. 新会话启动顺序

1. 先阅读本文、`CLAUDE.md:1-48`、`REGRESSION_ARCHIVE.md` Case 169-171。
2. 运行工作树基线命令，保留全部 dirty/untracked/deleted 状态。
3. 检查 `scripts/adjust_ctc_boundaries.py::_read_pause_intervals`、两个 resolver 定义、`pipeline_utils` span validator 和对应 tests。
4. 先写会失败的生产路径回归测试，再实施 R1-R4。
5. 按 mutation、tier rebuild、publication 顺序审查 `postprocess_textgrids.py`，完成 R5 和代表样本测试。
6. 实现 receipt fingerprints；禁止用历史无 fingerprint artifact 认证当前代码。
7. 聚焦测试、完整测试、compile 和 diff 检查全部通过后，再更新 regression archive。
8. 创建新的 v10 fresh 配置，冻结与 v7/v9 完全相同的 1000 stems。
9. 从头执行完整 route，不跳过任何会影响边界和 provenance 的阶段。
10. 扫描 canonical geometry、英文完整性、SP、标点、owner 和显示一致性。
11. 生成逐 stem v7/v9/v10 delta。
12. 只有 AC-01 至 AC-17 全部满足后，才能把任务状态更新为完成。
