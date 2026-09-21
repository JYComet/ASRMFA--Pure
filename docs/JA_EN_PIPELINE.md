# 日英 TTS 管线操作说明

这条管线与现有中文 MFA 管线分开运行。入口是
`scripts/run_ja_en_pipeline.py`，配置模板是
`configs/japanese_english_tts.yaml`。日语生产 phone 时间来自 Japanese MFA
v3.0.0，真英语来自 English US ARPA v3.0.0；Julius 只做可选诊断。

## 先理解 mora 和 phone

`さくら`（sakura）有 `さ｜く｜ら` 三个 mora（拍）。一个 mora 可以包含多个
phone，所以不能把三个拍直接当成三个 phone。`東京`（Tōkyō）按读音是
`と｜う｜きょ｜う` 四拍。长元音可以在不同表示中写成 `o:`、`o o` 或 MFA
词典中的 `oː`，这些是表示关系，不能用字符串替换或数组 `zip` 假造拍内的
声学边界。导出的 `mora_graph` 是多对多关系；没有 estimator 时不导出真实
mora 内部时间。

促音、拨音和清化元音也保留在 phone 关系中。低能量本身不会授权删除它们。
交付时保留原始文件和每个阶段的 JSONL、TextGrid、receipt；不把产物打 zip，也不把
一个 JSONL 拆成无法逐条重算的片段。每个 uid、unit、run 都必须在 manifest、raw MFA
区间、音频样本和输出集合中一一对应。
文本前端预测的重音写在 `accent_predicted`，音频测量的基频写在
`f0_measured`；两者有不同 provenance 和 mask，未知值保持未知。

## 配置、运行和恢复

必须显式指定 `pipeline`、`workspace`、`input_manifest`、供应链 lock，以及
前端 provider、commit 和全部布尔选项。`mixed.enabled` 和
`julius_diagnostic.enabled` 默认关闭，`julius_diagnostic.write_back` 必须为
`false`。生产发布要求供应链许可证、全部目标已解析、独立 verifier 通过。

```bash
python scripts/run_ja_en_pipeline.py --config configs/japanese_english_tts.yaml --check
python scripts/run_ja_en_pipeline.py --config configs/japanese_english_tts.yaml --inspect
python scripts/run_ja_en_pipeline.py --config configs/japanese_english_tts.yaml --stage inventory,audio,asr,reading,frontend,semantic,anchors,align,merge,tts,verify
python scripts/run_ja_en_pipeline.py --config configs/japanese_english_tts.yaml --resume --stage tts,verify
python scripts/verify_ja_en_tts.py --workspace /data/runs/ja_en_v1
python scripts/verify_ja_en_tts.py --workspace /data/runs/ja_en_v1 --integrity-only
```

`--check` 只验证 manifest、配置和路径，不加载 ASR/MFA。`--stage` 使用逗号
分隔的阶段名，`--resume` 会先重算 identity、文件集合和 receipt hash；源文件、
模型、配置、schema 或实现 digest 改变时 fail closed。每条记录的输出包括
`tts_training_records.jsonl` 和同 uid 的 TextGrid，后者固定包含 `words`、
`phones`、`language` 三个 tier；phone label 使用 `ja:` 或 `en:` 前缀。
多 token 记录使用 `selected_readings[token_id]` 与 `reading_evidence.locks[]`；只有单个
lexical token 才可使用 scalar `selected_reading`。每个 phone 同时保留 16 kHz alignment、
train 和 source 三个整数 sample axis（包括 `train_start_sample`/`train_end_sample`、
`source_start_sample`/`source_end_sample`），不能把 16 kHz 索引直接当作 48 kHz 训练切片。
verifier 默认只有 `release_ready=true` 才返回成功退出码；`--integrity-only` 只
适合离线开发检查，不能发布 production。

路径可以全部放在项目外的 run root，不要求固定磁盘布局：

```text
run-root/
  manifest.jsonl
  supply_chain_lock.json
  gold.json                 # 没有人工 gold 时只能 DEV/PARTIAL
  config.yaml
```

manifest 是 JSONL，每行至少有稳定 `uid`、`wav` 和 `text`：

```json
{"uid":"line-0001","wav":"audio/line-0001.wav","text":"さくら","speaker":"spk-a"}
```

上游 stage 注册 TTS 与 Julius adapter 后，TTS stage 会读取已准备好的
`alignment.jsonl`。每行必须绑定 `uid`、`train_wav`、`alignment_wav` 和
`ja-en-alignment-v2` 内容；缺任何绑定都会返回 `BLOCKED/REJECTED`。建议每次改动
模型、前端选项、字典、gold 或 manifest 都使用新的 workspace。

provider smoke 也使用同一份配置和新的 workspace：先运行 `--check`，再按依赖顺序运行
`--stage inventory,audio,asr,reading,frontend,semantic,anchors,align,merge,tts,verify`（只恢复已准备好的 TTS
输入时可用 `--resume --stage tts,verify`）；这些阶段只有在对应本地 runtime、模型 revision
和 receipt 已绑定时才会进入 `COMPLETE`。没有 runtime 的环境会保持 `BLOCKED`，
不会把接口检查当成真实 ASR/MFA 质量认证。

供应链 lock 使用 `verify_ja_supply_chain.py` 的实际格式。每个资源需要 `id`、`kind`、
`source_url`、固定 `commit`/`tag`/`revision`、`license` 对象和 `artifacts` 列表；严格生产
校验还要求每个 artifact 的路径和 SHA-256、许可证文件路径及哈希：

```json
{"schema":"ja-supply-chain-lock-v1","status":"frozen",
 "resources":[{"id":"japanese_mfa_dictionary","kind":"dictionary",
 "source_url":"https://example.invalid/source","revision":"v3.0.0",
 "license":{"status":"reviewed","path":"licenses/dict.txt",
 "file_sha256":"<64 hex>"},
 "artifacts":[{"id":"dict","path":"models/japanese.dict",
 "sha256":"<64 hex>"}],"artifact_hashes":[]}]}
```

人工 gold 的绑定格式是：

```json
{"target_id":"gate-20260921","sample_rate":16000,
 "rows":[{"id":"mix-0001","bucket":"ja_to_en","gold_seam_sample":12345,
 "gold_route":"ja,en"}]}
```

生产时 gate 文件通过 `gold_path` 和 `gold_sha256` 指向外部 gold；`reading_gold.json`
必须有 60 个唯一 uid（consistent 30、ambiguous 30），`pure_gold.json` 必须有 20 个
唯一 uid（日语 10、英语 10），mixed gold 必须有四个 bucket 各 10 个 uid。gate 的
`rows` 是实际运行结果，不能把 gold 嵌在 PASS receipt 中自证。

canary 结果必须使用相同 `target_id` 和完整唯一 `id` 集合；verifier 会根据预测
seam sample 与 gold sample 重新计算 MAE/P95，不接受 producer 直接提交的
`seam_error_ms` 或单独的 `status: PASS`。

## 模型选择和安装边界

`baseline_3family` 才是 production reading 的最小 ASR profile：Qwen、Whisper
和 ReazonSpeech 每家族最多一票。`qwen_only_dev` 只用于开发，不得发布高置信
reading；五模型扩展也不会给同一家族增加票数。pyopenjtalk-plus 是固定 commit
的候选文本前端，独立环境只能有一个 `pyopenjtalk` namespace provider。MFA
运行时、声学模型、词典、wheel 和 Julius binary/model 都要分别 pin、计算
SHA-256 并审核许可证；代码许可不能替代模型许可。缺少这些证据时状态是
`BLOCKED`，不能静默换模型或回退到旧前端。

## Julius 诊断和发布门禁

Julius adapter（若显式启用且完成外部审计配置）只读取同一份 16 kHz mono PCM16
alignment WAV 和锁定 reading，用 `-palign` 解析 `.lab`，产物写入独立 diagnostic
namespace。它不参与 ASR
reading vote，不覆盖 MFA 时间，也不写回 TTS JSONL；缺 binary、model、dict
或许可证时生成 `diagnostic_unavailable`，不触发 production fallback。

开发门禁需要 reading 一致集 30 条和歧义集 30 条；mixed gate 需要 40 条（四个
方向桶各 10），至少接受 34 条且每桶至少 8 条，seam MAE ≤80 ms、P95 ≤160 ms，
route error、clipping、跨语言 overlap 和错误 COMPLETE 必须为零。缺少人工 gold、
target ID 不匹配或只剩全拒收时 gate 不会通过。当前环境只有合成输入的 runtime
smoke；真实 corpus、人工 gold、Whisper/Reazon runtime、Julius 资产和许可证审核
仍未完成，因此不能声称 production GO。

Julius 要求显式 converter/JATTS commits、binary/model/dict 路径、每个资产的
hash/license，以及 16 kHz mono PCM16 frozen alignment WAV。当前 adapter 是隔离的
审计工作流；缺少已 pin 的 converter/JATTS 源、binary、model、dictionary、许可证或
非空 `.lab` 时只生成 `diagnostic_unavailable`，不会假装完成，也不会回写生产物。
独立 verifier 报告区分 `integrity_ok` 与 `release_ready`：
离线记录即使能够重算 JSONL、TextGrid、audio transform、phones、aliases 和 mora
关系，缺真实许可证、人工 gold、ASR family evidence、MFA raw receipt 或 gate 时仍为
`release_ready=false`、`status=BLOCKED`。
