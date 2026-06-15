# Chinese MFA Forced Alignment Pipeline

基于 Montreal Forced Aligner (MFA) 的中文强制对齐 pipeline，将 wav 音频与中文文本对齐，生成 5 层 TextGrid。

音频处理保持原始采样率（32kHz），MFA 阶段临时降采样到 16kHz 对齐后自动清理，最终输出 32kHz 音频 + 对应 TextGrid。

## 目录结构

```
chinese_mfa_pipeline/              # 项目（纯代码 + 模型，可移植）
├── config.yaml                    # 全局配置
├── environment.yml                # conda 环境（25 个包，精确版本）
├── requirements.txt               # pip 依赖表
├── setup_env.bat                  # Windows 一键安装
├── setup.sh                       # Linux/Mac 一键安装
├── scripts/
│   ├── run_pipeline.py            # 全流程编排
│   ├── prepare_corpus.py          # wav+txt 匹配 & 中文→拼音
│   ├── trim_silence_batch.py      # 静音裁剪 + 首尾补全
│   ├── postprocess_textgrids.py   # 后处理（5 层、质检、BGM 检测）
│   ├── view_in_praat.py           # Praat 查看器
│   ├── convert_dict_to_ipa.py     # 词典格式转换
│   └── group_audio.py             # 文件分组工具
├── dict/                          # 发音词典
└── models/mfa/                    # MFA 预训练模型

workspace/                         # 全部输出（项目外，配置中指定路径）
├── audio/                         # 裁剪 + 静音处理后的 WAV（原始采样率）
├── pinyin/                        # 中文→拼音文本
├── aligned/                       # MFA 原始对齐 TextGrid
├── output/                        # ★ 通过质检的 5 层 TextGrid + 报告
├── filtered/                      # 未通过质检的 TextGrid
├── validate/                      # MFA 验证输出
└── temp/                          # MFA 临时文件（含对齐日志，16k 副本自动清理）
```

## 新电脑部署

### 1. 安装 conda

下载 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 并安装。

### 2. 一键安装

```bash
# Windows
setup_env.bat

# Linux/Mac
bash setup.sh
```

自动完成：创建 `mfa_chinese` 环境 → 安装全部依赖 → 生成 IPA 词典 → 下载 MFA 模型。

### 3. 配置

编辑 `config.yaml`：

```yaml
workspace: D:/MFA_post_process/workspace   # 输出根目录（换电脑只改这一行）
data_dir: data_dir                          # 输入目录（也可用 --data-dir）
txt_suffix: qwen3-api                       # 匹配的 ASR 引擎
```

### 4. 准备数据

```
data/
└── 录音1/
    └── segments/
        ├── 录音1_seg001.wav
        ├── ...
        └── txt/
            ├── 录音1_seg001_qwen3-api.txt   # 原始中文文本
            └── ...
```

- 音频文件夹内任意嵌套层级均可自动识别
- 通过 `txt_suffix` 指定使用哪个 ASR 引擎的文本

### 5. 运行

```bash
conda activate mfa_chinese

# 完整运行
python scripts/run_pipeline.py --data-dir E:/path/to/data

# 跳过某步
python scripts/run_pipeline.py --data-dir E:/path/to/data --skip-trim

# 只跑一步
python scripts/run_pipeline.py --data-dir E:/path/to/data --step align --overwrite
```

## Pipeline 步骤（6 步）

| # | 步骤 | 说明 | 输出 |
|---|------|------|------|
| 1 | trim | 裁剪内部静音（>1s→1s）、首尾补静音（0.5s） | workspace/audio/（原始采样率） |
| 2 | prepare | 匹配 wav↔txt、中文→拼音 | workspace/pinyin/ |
| 3 | resample | 降采样到 16kHz 放入临时目录（供 MFA 使用） | workspace/temp/audio_16k/ |
| 4 | validate | MFA 语料验证 | workspace/validate/ |
| 5 | align | MFA 强制对齐 | workspace/aligned/ |
| 6 | postprocess | 5 层 TextGrid 构建 + 修正 + BGM 检测 + 质检过滤 | workspace/output/ + filtered/ |

> 步骤 3 自动执行，16kHz 临时文件在流水线结束后自动清理。

## Postprocess 详解（Step 6）

### 阶段 1：5 层 TextGrid 构建

从 MFA 输出的 2 层（words + phones）扩展为 5 层：

| 层 | 来源 |
|----|------|
| raw_text | 从源目录递归搜索原始中文文本 |
| pinyin | 中文→拼音，标点映射为 `[PAUSE]` / `<PAUSE>` |
| words | MFA 对齐的音节（拼音+声调）+ 静音标记 |
| phones | MFA 对齐的音素（IPA 符号）+ 静音标记 |
| pinyin_phones | IPA 音素反向映射为拼音声韵母（通过词典拆分声母+韵母） |

### 阶段 2：修正

- **静音合并** — 持续 <0.2s 且能量接近前一个音素（非零均值 > 前音素 × 0.5）的短静音合并到前一音素
- **短词修正** — 功能词（`的`、`了`、`de5`、`le5`、`zhe5` 等 <0.25s 的词）若前后有长静音，向后搜索语音起始点扩展词边界

### 阶段 3：BGM 检测

- 统计全局噪声基底（所有静音段底部 10% RMS 中位数）
- 逐个检查静音段：能量 > 噪声基底 × 2.0 且达到语音水平 → 标记 `bgm_suspect`
- 被标记的文件路由到 `filtered/`

### 阶段 4：质检过滤

| 规则 | 当前阈值 | 说明 |
|------|----------|------|
| short_phone | < 0.005s | 音素过短（低于一帧精度） |
| long_word | > 1.5s | 音节过长（可能的对齐错误） |
| low_phone_coverage | < 25% | 词内音素覆盖不足 |
| large_edge_gap | > 0.35s | 词-音素边界间隙过大 |
| short_word_between_silences | < 0.12s 且两侧静音 > 0.4s | 孤立短词（可能对齐错误） |
| sp3 | >= 1.5s 静音 | 过长停顿（说话人长时间沉默） |

所有静音按最终时长重标：`<sp0>`(<0.2s) `<sp1>`(<0.5s) `<sp2>`(<1.5s) `<sp3>`(\>=1.5s)。

通过的 TextGrid 写入 `output/` + `postprocess_report.jsonl`，未通过的写入 `filtered/`。

## 输出 TextGrid 结构（5 层）

| 层 | 内容 |
|----|------|
| raw_text | 原始中文句子 |
| pinyin | 拼音 + 暂停标记 [PAUSE] / \<PAUSE\> |
| words | MFA 对齐的音节（拼音+声调）+ 静音 \<sp0\>~\<sp3\> |
| phones | MFA 对齐的音素（IPA 符号）+ 静音 |
| pinyin_phones | IPA 音素反向映射为拼音声韵母 |

静音分级：`<sp0>`(<0.2s) `<sp1>`(<0.5s) `<sp2>`(<1.5s) `<sp3>`(>=1.5s)

## 配置参考

| 配置项 | 说明 |
|--------|------|
| `workspace` | 输出根目录 |
| `data_dir` | 输入目录 |
| `txt_suffix` | ASR 引擎后缀 |
| `trim.sil_vol_threshold` | 静音检测灵敏度（越大越激进） |
| `trim.target_sr` | 输出采样率（null=保持原始） |
| `postprocess.filter_short_phone_sec` | 最短音素（快语速调低） |
| `postprocess.filter_long_word_sec` | 最长词（慢语速调高） |
| 更多... | 见 `config.yaml` 注释 |

## CLI 参数

| 参数 | 说明 |
|------|------|
| `--data-dir PATH` | 输入目录 |
| `--output-dir PATH` | 输出目录 |
| `--config PATH` | 配置文件 |
| `--step NAME` | 只运行某步 |
| `--skip-to NAME` | 从某步开始 |
| `--force` | 遇错继续 |
| `--overwrite` | 覆盖已有输出 |
