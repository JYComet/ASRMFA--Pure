# docs/archive — 历史文档归档

这里的文档**不是当前指导**，只是留档。它们记录了当时的事实、当时的判断和当时的路径。

## 什么会进这里

- 一次性任务的交接文档（`handoffs/`）——任务结束后不再维护
- 已完成或已放弃的计划（`plans/`）
- 阶段性状态报告、审计报告、专题分析

## 当前有效的东西在哪里

| 想知道 | 看这里 |
|--------|--------|
| 某个 bug 的现象／根因／修复／验证 | `REGRESSION_ARCHIVE.md`（唯一持续维护的记录） |
| 怎么用这条管线 | `README.md` |
| Qwen3 接入的能力边界与运行条件 | `docs/QWEN3_UPGRADE_ANALYSIS.md`、`docs/QWEN3ASR_MODE.md` |
| 还没做的待办 | `handoffs/LARIA-v3-follow-up.md`（唯一留在 `handoffs/` 的） |

## 两条阅读须知

1. **文档里的路径是当时的样子，没有随仓库整理而改写。** 例如文中出现的
   `configs/xxx.yaml` 可能已移到 `configs/archive/{2026-08,2026-09,undated}/`，文中
   引用的根目录级脚本 `check_ipa_mapping.py`、`verify_risks.py` 已移入 `scripts/`，
   其中的 `compileall -q scripts check_ipa_mapping.py verify_risks.py` 一类命令按今天的
   布局应当写作 `compileall -q scripts`。归档文档不改写，改了就变成伪造历史。

2. **这里是归档，不是"覆盖关系"的证明。** 下面这几份是**因为未被
   `REGRESSION_ARCHIVE.md` 覆盖**才归档而非删除的，它们含有存档里没有的内容：

   - `FILTER_ANALYSIS_REPORT.md` —— 2026-07-13 对 100,929 个文件／161 个数据集的过滤
     原因统计（`word_in_silence` 2,686 条／94.0% 等）。曾按 Case 16 归类，
     但 Case 16 讲的是 MFA `--fine_tune` 默认值，不是同一主题。
   - `CROSS_CASE_ANALYSIS.md` —— Cases 1–13 的综合视图，含「六大共同根因」
     「案例全景矩阵」「Case 间直接因果链」。该文第 240 行还指出
     `FILTER_ANALYSIS_REPORT` 中 `mid_sp` 的子原因 (a)「长停顿无标点」与
     (c)「锚点错位」**尚未**在存档中有系统性修复。

   另：`CROSS_CASE_ANALYSIS.md` 开头把 `hanzi-tier-bugfix.md` 列为分析范围，而该文件
   已被删除——其内容由 `REGRESSION_ARCHIVE.md` 的 Case 10 覆盖。
