# MFA 管线排查与修复（2026-09-07）

## 流程与数据职责

主管线由 `scripts/run_pipeline.py` 编排，批量任务由
`scripts/streaming_pipeline.py` 调度。默认完整流程为：

```text
trim → resample → prealign
  → normalize_punct → normalize → normalize_ria → normalize_en
  → adjust → align → align_en → postprocess → strict_ok
```

- `trim/resample` 准备音频及 MFA 使用的 16 kHz 音频。
- `prealign` 产生 CTC 文本、`.lab`、词边界 TextGrid 和时间轴/来源收据。
- 规范化与 `adjust` 在派生工作目录中更新文本和边界，原始 CTC 产物保留作为来源证据。
- `align` 优先消费调整后的 CTC 语料，检查冻结的 stem 集合、音频、锚点及时间轴，再运行中文 MFA。
  语料通过 `.lab` 提供，锚点通过定制参数 `--textgrid_directory` 提供。
- 大语料的 `_run_mfa_sharded` 创建独立分片语料、音频、锚点与 MFA 工作目录；
  小语料走单进程。缺失输出进入按 stem 重试及更宽 beam 的救援流程。
- `align_en` 使用 `scripts/align_english_mfa.py`，按英文 CTC 片段裁剪音频，
  用英文词典和声学模型独立对齐，再输出 phones/ledger/manifest 供后处理核验。
  它不依赖中文分支的 `--textgrid_directory` 参数。
- `postprocess` 结合 MFA 音素、CTC 边界、英文证据和 NVV 信息生成五层 TextGrid；
  `strict_ok` 独立核验后才形成最终可发布结果。

`ctc_ready` 从已有 CTC 产物接入；无参考模式先执行 `pad_silence/prealign`。
`validate` 是可选单步，不在当前默认完整步骤序列中。

## 已修复问题

### 1. MFA 静默忽略 CTC 锚点参数

**根因**：主管线假设成功调用 MFA 就代表锚点被使用。但本机 `mfa-dev` 的
MFA 3.3.9 允许未知 CLI 参数，并且会丢弃不认识的配置项。
实际执行 `PretrainedAligner.parse_parameters(None, {},
['--textgrid_directory', '/tmp/anchor-proof'])` 返回 `{}`。
当 `.lab` 与 TextGrid 同名时，普通语料解析还会优先选择 `.lab`，因此不能依靠
把锚点文件放在语料旁边来补偿这个参数缺失。

**修复**：新增 `pipeline_utils.require_mfa_anchor_support`，用选定的 Python
和 MFA 环境启动小型探针，检查实际参数解析结果是否保留锚点参数。主入口、
直接 MFA 调用、分片入口及单进程重试都执行必要检查；不支持、导入失败、
超时或无法验证时明确失败。无锚点的调用不受此检查限制。

这修复的是静默降级与错误成功判定；没有把原版 MFA 改造成支持该定制锚点协议的实现。
本机 `mfa-dev` 已通过真实探针确认会被拒绝。实际锚定对齐仍需通过
`python_path`/`--python` 选择支持该协议的 MFA 环境。

### 2. MFA 输入检查失败前删除旧结果

**根因**：`step_mfa_align` 原先先执行 `--overwrite` 清理，再检查语料集合、
缺失音频、缺失锚点和时间轴。即使新任务无法开始，旧 TextGrid 也已删除。

**修复**：把输出目录及旧 SQLite 数据库的清理移动到运行时、输入集合、
时间轴和严格目标目录检查之后。失败的预检保留旧结果；清理错误也不再被静默吞掉。

## 验证与范围

- 修复前：已有测试 `1024 passed`；新增用例复现了错误启动、旧结果被删及失败后仍暂存分片/重试目录。
- 修复后：`PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests`
  得到 `1031 passed`。
- 新增用例使用真实子进程和隔离的模拟 MFA 包，覆盖参数被消费/被丢弃、无锚点调用、
  预检保留旧结果及分片/重试提前退出。
- 另用本机真实 MFA 3.3.9 验证不兼容运行时被明确拒绝。
- 未执行 GPU/全量语音对齐，未修改声学模型、生产数据或已安装的 MFA。
  这些验证不构成声学对齐质量的端到端验收。

本次保留了开始时已有的配置、词典、CTC 代码和测试改动；异常记录追加在
`REGRESSION_ARCHIVE.md` 的 Case 245–246。

## reverse1999 重跑记录（2026-09-09）

按 8 卡重跑后，CTC prealign 完成 39,645 条有效、175 条过滤（总计 39,820，
`silent_loss=0`）。后续 MFA anchor 对齐未启动：当前 `mfa-dev` 运行时不支持
`--textgrid_directory`，流水线安全拦截并发布 0 条，因此该次重跑没有作为最终
发布来源；保留新 workspace 及 CTC 收据作为审计证据。

最终从已完整结束的旧批次
`/mnt/nvme3/mfa_work_gamedata_reverse1999_20260903/output_staging/20260904T065249Z_3677774_3677774`
恢复了 11,923 条 `ok` 对齐结果。报告共 39,644 条，其中 27,718 条按既有规则过滤，
另有 3 条 `legacy_or_unprovenanced_nvasr_row` 错误和 1 条缺失 MFA 对齐；后 4 条占
总报告的 0.010090%，按“异常占比小则不重跑”的要求排除，不再发起重跑。

全部游戏的可用结果随后按原始音频路径的直接父目录重新归入说话人目录。最终发布
151,877 条 TextGrid 及一一对应的补齐前后静音 WAV；reverse1999 的 11,923 条 WAV
均生成了 0.5 秒前置及后置静音版本，对应 TextGrid 时间轴同步平移。分类收据位于各
游戏输出目录的 `.speaker_classification_receipt.json`。
