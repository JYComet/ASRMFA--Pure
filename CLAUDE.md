# CLAUDE.md — MFA Pipeline 项目配置

## 异常存档规则

当修改 `scripts/` 下的代码以修复一个由逻辑冲突导致的 bug（非简单笔误）时，
必须在修复后将异常记录写入 `REGRESSION_ARCHIVE.md`：

### 触发条件（满足任一即写入）

- 两个处理步骤对同一段数据做出矛盾决策（如 A 切掉了静音，B 又合回去了）
- 某条规则/阈值产生了误判（如把切尾静音误判为"词被压缩"）
- `kind` / 标记不匹配导致某段逻辑被跳过
- 处理顺位（phase ordering）问题：A 步骤的输出被 B 步骤意外覆盖

### 写入内容

按 `REGRESSION_ARCHIVE.md` 末尾的模板格式追加，至少包含：
- 现象（修复前 vs 修复后的边界/数值对比）
- 根因链（按步骤追溯）
- 涉及的函数和行号
- 验证方法（可复现的检查代码）

### 不触发的情况

- 纯语法错误、typo、import 缺失
- 参数调优（阈值、权重）
- 新增功能（非修复）

## 项目路径

- 项目根: `E:\MFA_Pause\repo`
- 管线入口: `scripts/run_pipeline.py`
- 后处理: `scripts/postprocess_textgrids.py`
- CTC 边界修正: `scripts/adjust_ctc_boundaries.py`
- 异常存档: `REGRESSION_ARCHIVE.md`

## 常用命令

```bash
# 运行 ctc_ready 管线（跳过 trim + prealign）
python scripts/run_pipeline.py --config configs/<name>.yaml

# 只跑 postprocess
python scripts/run_pipeline.py --config configs/<name>.yaml --step postprocess --overwrite

# MFA Python 路径
/c/Users/MSI/miniconda3/envs/mfa/python
```
