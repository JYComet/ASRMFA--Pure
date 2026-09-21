# 日英 MFA 五轨与逐音素重音设计

## 目标

在现有日英 ASR → reading → frontend → MFA → TTS 管线中增加可追溯的五轨交付格式，并修正日语 mora、MFA 原生音素、TTS 基本音素与重音标签之间的关系。

最终每条已接受记录必须输出五条 TextGrid 轨道：

1. `original_text`：原始台本文本；
2. `kana`：锁定后的日语假名 reading；
3. `mfa_phone`：Japanese MFA 或 English US ARPA 的原生音素；
4. `phone_kana`：依据显式 mora 映射投影到每个 MFA 音素区间的假名；
5. `phone_tone`：依据显式 mora 映射投影到每个 MFA 音素区间的离散 tone。

TextGrid 是便于人工检查的交付视图。JSON 关系图是结构、来源、掩码、时间轴和校验的权威数据源。

## 不变量

- MFA 是唯一生产原生 phone timing 后端。Qwen ForcedAligner 继续只提供 lexical anchors。
- Japanese MFA 与 English US ARPA 使用独立模型、inventory、词典和运行目录。
- reading 必须先按 occurrence 锁定；accent 或 tone 阶段不得重新选择 reading。
- mora 与 MFA 原生 phone 是多对多关系。
- MFA 原生 phone 的边界不得通过平均切分、F0 拐点或最近邻规则改写。
- 文本重音先验与音频 F0 测量使用不同字段、来源和 mask。
- 未知 tone 保持 `UNK`；英语、静音和非语言事件使用 `NA`。
- 禁止最近邻填充 H/L，禁止将“无 F0”转换为 `L`。
- 第 3、4、5 轨的区间数量、`xmin` 和 `xmax` 必须逐项完全相同。
- 每个最终 phone 必须追溯到原始 MFA TextGrid interval、唯一 occurrence alias、唯一词典发音和整数 sample 轴。

## 选定架构

采用双层音素模型。

### 原生 MFA 音素层

`native_phone` 保存模型真实输出：

- Japanese MFA 或 English US ARPA label；
- 原始 TextGrid interval ID；
- alignment、training 和 source 三条整数 sample 轴；
- 所属 language、unit、alias 和 run；
- 对应的零个或多个 mora；
- 对应的一个或多个 basic phone；
- 从 basic phone 合并到 native phone 的变换类型。

这层的时间是权威声学边界。

### TTS 基本音素层

`basic_phone` 表达可用于 TTS 条件和 tone 监督的结构：

- 每个日语 basic phone 恰好属于一个 mora；
- 普通拍首辅音和元音可以共享同一 mora；
- 长元音展开为两个 basic vowel；
- 促音长辅音展开为 `Q + onset`；
- `ン + 鼻辅音` 合并展开为 `N + nasal onset`；
- 清化保留 basic vowel 并设置 realization/mask；
- 脱落保留 basic phone，但允许没有 native phone 和时间。

`Q` 和 `N` 只属于 semantic/TTS 层，绝不写入 Japanese MFA 词典，除非未来更换为原生支持它们的声学模型。

基本音素层不自动拥有独立时间。若多个 basic phone 对应同一 native phone，默认只保存 group 总时长约束。

## 五轨 TextGrid 合同

### 1. `original_text`

- 粒度为最终 lexical unit/word。
- label 来自原始台本的精确 source span，不使用 normalized surface 替代。
- 无独立声学边界的标点和空白以 display-only 方式附着到相邻 lexical unit；JSON 保留原始 span 和附着方向。
- 非空 label 按 source span 顺序拼接后，必须在声明的 display normalization 下复原原文。

### 2. `kana`

- 与 `original_text` 使用相同 lexical unit 边界。
- 日语 label 使用锁定后的 kana reading。
- 英语、静音和非语言区间为空字符串；不得为英语制造日语假名。
- reading override 后只能输出 override 绑定的 kana，不能退回 frontend 默认 reading。

### 3. `mfa_phone`

- 粒度为 merged native MFA phone timeline。
- 日语使用 `ja:<native_phone>`，英语使用 `en:<native_phone>`。
- 原始 MFA 空白/静音区间在规范化 timeline 中保留为空或声明的 silence event；不得制造语言 phone。
- ARPABET stress digit 保留在 native label 中，但不转换成日语 tone。

### 4. `phone_kana`

- 与 `mfa_phone` 逐区间同构。
- 单 mora 映射输出该 mora 的 kana。
- 跨 mora native phone 按时间前后顺序用 `|` 连接，例如 `コ|ー`、`ン|ナ`。
- 英语、静音和非语言事件输出空字符串。
- label 必须仅由 JSON 中 `native_phone.mora_ids` 和 mora nodes 派生；禁止按 phone 字符串、数组下标或最近 mora 猜测。

### 5. `phone_tone`

- 与 `mfa_phone` 逐区间同构。
- 单 mora 映射输出 `H`、`L` 或 `UNK`。
- 跨 mora native phone 按相同 mora 顺序用 `|` 连接，例如 `H|L`。
- 英语、静音、呼吸、笑声和其他非日语 lexical event 输出 `NA`。
- 轨道值是文本/人工的 mora tone 投影，不是 phone 区间的 F0 测量结论。

### 区间覆盖

五个 IntervalTier 共享同一 `xmin=0` 和 `xmax=alignment_frames/sample_rate`。writer 为词间和音素间未标注区域生成空 label 区间，保证合法的连续 IntervalTier。verifier 在比较 lexical 或 phone cardinality 时只计算带稳定 segment ID 的权威区间，不把补齐覆盖面的空区间当成新 phone。

## Schema

新生产入口使用以下版本：

- `ja-semantic-phone-graph-v2`
- `ja-en-alignment-v3`
- `ja-prosody-alignment-v1`
- `tts-training-record-v2`
- `five-track-textgrid-v1`

旧 v1/v2 artifact 只允许历史读取和明确迁移；新入口不得继续生成旧版本。迁移不能凭旧 TextGrid 推断缺失的 basic-phone role 或 tone。

### Mora node

每个 mora node 至少包含：

```json
{
  "mora_id": "u1:tok1:mora_0001",
  "token_id": "tok1",
  "kana": "ン",
  "kind": "nasal_mora",
  "mora_index": 1,
  "accent_phrase_id": "ap0",
  "tone": "H",
  "tone_known": true,
  "tone_source": "contextual_frontend_prediction"
}
```

`kind` 的封闭集合是：

- `regular`
- `long_extension`
- `sokuon`
- `nasal_mora`
- `final_sokuon`
- `devoiced`
- `elided`

### Basic phone node

每个 basic phone node 至少包含：

```json
{
  "basic_phone_id": "bp3",
  "symbol": "N",
  "role": "nasal_mora",
  "mora_id": "u1:tok1:mora_0001",
  "realization": "merged",
  "tone": "H",
  "tone_known": true
}
```

允许的 `role` 至少包含 `onset`、`nucleus`、`long_extension`、`sokuon`、`nasal_mora`、`final_sokuon`。`realization` 使用 `observed`、`merged`、`devoiced`、`elided` 或 `unresolved`。

### Native phone node

每个 native phone node 至少包含：

```json
{
  "phone_id": "p17",
  "native_phone": "nː",
  "language": "ja",
  "start_sample": 12400,
  "end_sample": 13920,
  "raw_interval_id": 8,
  "mora_ids": ["mora_0001", "mora_0002"],
  "basic_phone_ids": ["bp3", "bp4"],
  "transform": "nasal_coalescence",
  "boundary_source": "mfa_native_interval"
}
```

`transform` 的封闭集合是：

- `identity`
- `long_vowel_merge`
- `geminate_merge`
- `nasal_coalescence`
- `devoiced_realization`
- `final_sokuon`

英语 native phone 的 `mora_ids` 和 `basic_phone_ids` 可以为空，但必须保留 language、run、alias 和 raw interval provenance。

`elided` basic phone 没有对应的 native phone node；它通过 basic node 的 `realization=elided`、空时间和 mora coverage 保存，不能伪造一个零时长 native interval。

### Duration group

跨 basic phone 的 native interval 使用：

```json
{
  "duration_group_id": "dg17",
  "native_phone_id": "p17",
  "basic_phone_ids": ["bp3", "bp4"],
  "total_duration_samples": 1520,
  "internal_boundaries_known": false,
  "boundary_source": "unknown_inside_mfa_interval",
  "duration_loss_mode": "group_sum"
}
```

若未来启用 estimator，必须额外记录模型 identity、confidence 和 `boundary_source=estimated`。在专门验收通过前，独立 basic duration 的 `duration_loss_mask` 必须为 false。

## Tone 生成

### 来源优先级

1. 绑定 uid/token/mora 和 locked-reading digest 的人工 tone override；
2. 绑定版本、条目和资源哈希的固定 accent lexicon；
3. 固定 provider、模型/commit、flags 和 full-context evidence 的 frontend prediction；
4. 无可靠来源时为 `UNK`。

不同来源不得按投票或平均混合。较高优先级覆盖较低优先级，并保留被覆盖来源的审计记录。

### Frontend 证据

当前单个 `accent_nucleus` 字段不足以生成 phrase-level tone。frontend contract 必须增加：

- `accent_phrase_id`
- `mora_index_in_phrase`
- accent phrase 边界
- nucleus/downstep 位置
- 可重算的 full-context label 或等价结构化字段
- provider、revision、flags 和模型 digest

从 phrase accent 到逐 mora H/L 的转换必须由固定版本的 adapter 完成，并用人工 fixture 验证平板、头高、中高、尾高和多词 accent phrase。

当 locked reading 不等于原 contextual reading 时，原 accent prediction 失效。第一版只有人工 override 或绑定 locked reading 的固定 accent lexicon 可以恢复已知 tone；不得通过对孤立 kana 再跑 frontend 冒充原句上下文重音。

### Tone 与 F0

权威字段保持分离：

```text
mora_tone_prior: H | L | UNK
phone_tone_projection: H | L | H|L | UNK | NA
f0_measured_hz: finite number | null
f0_observed: boolean
```

辅音继承所属 mora 的 tone 是训练条件广播，不表示辅音本身存在可靠基频。清化、促音闭塞和无声区间可以 `tone_known=true` 且 `f0_observed=false`。

## 特殊音系规则

### 普通 CV、拗音、破擦音

同一 mora 的 onset 和 nucleus 各自成为 basic phone，并共享 mora/tone。`tɕ`、`dʑ`、`mʲ` 等按 MFA inventory 中的完整 label 处理，不能按 Unicode 字符拆分。

### 长元音

例如 `コー`：

```text
mora:         コ        ー
basic_phone:  k   o     o
native_phone: k   oː
```

`oː` 关联两个 mora 和两个 basic vowel。TextGrid `phone_kana` 为 `コ|ー`，`phone_tone` 可以为 `H|L`。内部两个 `o` 的时间默认未知。

### 促音长辅音

例如 `キット` 中 `tː` 映射为 `Q + t`：`Q` 属于 `ッ`，`t` 属于 `ト`。塞音、擦音、破擦音和浊辅音可以共享结构合同，但不能共享一条未经验证的声学切分规则。

### 拨音与长鼻音

仅当 locked reading 和选定 pronunciation 证明存在 `ン + 后续鼻辅音` 时，`mː`、`nː`、`ɲː` 等才使用 `nasal_coalescence`，展开为 `N + onset`。不得仅凭 `ː` 判断促音或拨音来源。

### 清化与脱落

清化保留 mora/basic vowel，设置 `realization=devoiced` 和 `f0_observed=false`。允许的元音脱落保留 mora/basic phone，设置 `realization=elided`、`native_phone_id=null`、时间为空；不得从邻接 phone 切出伪区间。无法证明是允许变体时，记录进入 unresolved/rejected，而不是由 prosody 阶段修复。

### 表现性延长、句末促音和非语言事件

表现性延长不得按时长自动增加 mora。句末 `ッ` 仅在 locked reading 明确包含该 mora 时使用 `final_sokuon`。呼吸、笑声和非语言事件不关联 mora，tone 使用 `NA`。

### 英语

English US ARPA phone 保留原生 label 和 stress digit。英语不建立日语 mora，不将 ARPABET stress 转换为 H/L；`phone_kana` 为空，`phone_tone=NA`。

## 管线和缓存

生产阶段顺序修改为：

```text
inventory → audio → asr → reading → frontend → semantic
          → anchors → align → merge → prosody → tts → verify
```

`prosody` 读取 locked reading、frontend full-context evidence、semantic v2 graph、merged native phone timeline 和人工/词典 tone 资源，输出 `ja-prosody-alignment-v1`。它不得修改 phone boundary、reading、alias、词典或 MFA artifact。

将 prosody 放在 merge 之后，使 tone 规则或人工 accent override 的变化只使 `prosody → tts → verify` 缓存失效，不触发 ASR 或 MFA 重跑。

prosody cache identity 必须绑定：

- locked-reading artifact 和 digest；
- frontend provider/build/options/full-context evidence；
- semantic adapter implementation hash；
- merged alignment artifact 和 raw MFA provenance；
- manual tone override/lexicon path、hash 和 schema；
- tone projection algorithm version；
- TextGrid schema version；
- Python/runtime identity。

旧 workspace 中 v1 schema 的 resume 必须返回 `resume_identity_drift` 或 `resume_stale`。恢复测试使用新的 workspace，保留原失败 workspace 作为证据。

## 错误和分区

新增稳定错误码：

- `accent_phrase_unresolved`
- `tone_cardinality_mismatch`
- `native_basic_mapping_ambiguous`
- `phone_tone_projection_lossy`
- `five_track_boundary_mismatch`
- `tone_provenance_missing`

单个 unit 的 mapping/tone 失败必须精确进入 unresolved/rejected ledger，不得让其他 UID 丢失账本。开发配置允许输出 `UNK`；当 `publish.require_all_tones=true` 时，任何已接受日语 mora 的 `UNK` 都阻止 release。英语 `NA` 不计作未知 tone。

## 独立校验

verifier 必须从外部 artifact 重开并重算：

- TextGrid 恰好包含五个规定 tier，名称和顺序固定；
- 第 3、4、5 轨的权威 segment ID、数量、整数 sample 边界逐项相同；
- `phone_kana` 仅由 native phone 的 ordered `mora_ids` 生成；
- `phone_tone` 仅由同一 ordered mora list 的 tone 生成；
- native pronunciation 顺序同时匹配 occurrence alias、locked dictionary、raw MFA TextGrid 和 merged alignment；
- 每个日语 native speech phone 至少关联一个 basic phone 和一个 mora；
- 每个日语 basic phone 恰好关联一个 mora；
- 每个日语 mora 被 native phone 覆盖，或显式标记为 `elided`；
- pure English 记录不声明日语 mora/tone；
- 所有已知 H/L 具有可重开的 provenance；
- estimated internal boundary 不得声明 `boundary_source=mfa_native_interval`；
- 五轨 TextGrid、JSONL、receipt 和 stage identity 的 hash 一致。

任何 producer 自填的汇总字段都不能替代 verifier 对外部文件的重算。

## 测试与验收

### 单元 fixture

- `さくら`：普通多 phone → 单 mora；
- `東京 / トウキョウ`：两个长元音跨 mora；
- `キット`：`tː → Q + t`；
- `オンナ`：`nː → N + n`；
- `ウンメー`：长鼻音与长元音同时存在；
- `グッズ`：使用 Japanese MFA 支持的 `dzː`，不得生成 `zː`；
- `スキ`：清化与允许的脱落；
- 句末 `ッ`；
- 日英 mixed：日语 H/L、英语 `NA`、五轨同边界；
- reading override：旧 accent 失效并得到 `UNK` 或人工新值。

### 负向测试

- 调换 mora 顺序；
- 删除 basic phone role；
- 把 `nː` 错标为 geminate；
- 为英语写入 H/L；
- 为 `oː` 折叠冲突 tone 为单值；
- 修改任一 `phone_kana` 或 `phone_tone` label；
- 第 4/5 轨边界偏移一个 sample；
- 篡改 raw interval ID、alias、词典或 provenance；
- 对未知 tone 使用最近邻填充。

以上情况都必须由 schema validator 或独立 verifier 拒绝。

### 集成验收

1. 在 fresh workspace 完成真实 pure JA、pure EN 和 mixed CLI 全链。
2. `integrity_ok=true`，但缺生产 gold/许可证时仍保持 `release_ready=false`。
3. resume 不重算，全部输出 hash 稳定。
4. 修改 tone override 只重跑 prosody、tts、verify。
5. 修改 MFA 模型/词典/semantic mapping 时使 align 及所有下游 artifact 失效。
6. Julius 开关不改变 production 五轨 TextGrid 或 TTS JSONL hash。
7. 最终运行完整 pytest、编译检查、受保护文件 hash 检查和 fresh CLI tamper 回归。

## 文件边界

预计新增：

- `scripts/ja_prosody.py`：mora tone、native/basic projection、prosody receipt；
- `tests/test_ja_prosody_projection.py`：特殊音系和 tone 投影；
- `tests/test_ja_five_track_textgrid.py`：五轨 writer/verifier 合同。

预计修改：

- `scripts/ja_en_schema.py`
- `scripts/ja_frontend.py`
- `scripts/ja_phone_adapter.py`
- `scripts/ja_en_stage_inputs.py`
- `scripts/merge_ja_en_mfa.py`
- `scripts/ja_tts_export.py`
- `scripts/verify_ja_en_tts.py`
- `scripts/run_ja_en_pipeline.py`
- `configs/japanese_english_tts.yaml`
- `docs/JA_EN_PIPELINE.md`
- 对应的 foundation、frontend、mora、stage-input、export、verifier、resume 和 mixed 测试。

不修改旧中文默认 pipeline 的阶段顺序、normalization、phone map 或发布入口。
