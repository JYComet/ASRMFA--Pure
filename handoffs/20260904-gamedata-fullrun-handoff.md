# GAMEDATA 全量对齐任务交接（2026-09-04）

> 2026-09-09 的四游戏增量重建、说话人分类、GAMESL 静音补全和最终发布结果，
> 见 `handoffs/20260909-gamedata-four-game-rebuild.md`。该文档的结果状态覆盖本文中
> 关于 reverse1999 和后续增量数据的旧状态。

## 一、初始任务

用主管线处理 `\\RS3621\Research_TTS\Data\Raw\GAMEDATA` 下数据，按游戏文件夹分类：
- 有参考文本 → 有参考文本模式；无参考文本 → 无参考文本模式
- 重返未来1999：忽略参考文本，用无参考文本模式重新生成
- 尘白禁区：后缀带 `jp` 的是日语音频，删除音频+对应文本
- ogg 转 wav
- 八卡显存不足 → 先传数据到 NVMe，等显存足够再通知继续

## 二、已完成

### 数据准备（全部完成）
- 源：`/mnt/nas/Research_TTS/Data/Raw/GAMEDATA`（=`/mnt/Raw/GAMEDATA`，同一 inode）
- 8 游戏：原神 genshin / 尘白禁区 snowbreak / 异环 yihuan / 环行旅舍 huanxing /
  白荆回廊 baijing / 终末地 zhongmodi / 绝区零 zzz / 重返未来1999 reverse1999
- **已删除日语**（NAS 源，清单在 `/mnt/nvme3/gamedata_20260903/.manifests/`）：
  - 尘白 `_jp`：508 文件；环行旅舍 `_ja`：4372 文件
- 异环 ogg→wav：11389 个（用 `/home/user/miniconda3/envs/mfa-dev/bin/ffmpeg`）
- 落地 NVMe：`/mnt/nvme3/gamedata_20260903/<codename>/`，104GB，236,849 wav
- 白荆回廊：无参考文本(0 txt)→无参考模式；剔除 444 个 `__hash` 副本
- 环行旅舍：stem 加角色前缀（通用台词名会碰撞）
- 重返未来1999：只拷 wav（无 txt）→无参考模式
- 异环/环行旅舍/终末地/绝区零/原神：有参考模式
- 落地脚本：`scripts/stage_gamedata.py`

### 配置（已建并校验，`configs/gamedata_*_20260903.yaml`）
- 6 个 `_reference_`（authority）、2 个 `_noref_`（fallback）
- ASR 双卡：`CUDA_VISIBLE_DEVICES=5,6` + `ctc_prealign.all_gpus: true`
- MFA 全 CPU：`mfa.num_jobs: 64` / `mfa_en.num_jobs: 16`
- `postprocess.strict_ok: false`（跳过尚在开发中的独立 strict_ok 审计）
- 无参考 `mfa_en.strict_provenance: false`（fallback ASR 文本英文 token 无法严格建词典）

### 已发布（5/8，输出 `/mnt/Raw/GAMEDATA_对齐_20260903/<codename>/`）
| 游戏 | 模式 | output | filtered |
|---|---|---:|---:|
| 尘白禁区 snowbreak | 参考 | 1221 | 292 |
| 环行旅舍 huanxing | 参考 | 1910 | 276 |
| 白荆回廊 baijing | 无参考 | 1421 | 4129 |
| 终末地 zhongmodi | 参考 | 7815 | 1376 |
| 异环 yihuan | 参考 | 4894 | 2260 |

## 三、当前进行中（最后 3 游戏）

已用 `setsid nohup bash /tmp/run_rest3.sh` 脱离会话重启（防止会话切换杀进程）。
运行日志：`/tmp/run_rest3_detached.log`；游戏日志：`/tmp/run_<codename>.log`。顺序：
`重返未来1999 reverse1999(noref) → 绝区零 zzz(reference) → 原神 genshin(reference)`
- 日志：`/tmp/run_<codename>.log`
- 完成后运行器打印 `REST3 COMPLETE`，监控 `b18dca2zj` 会在每个游戏 DONE 时通知。

## 四、修复的 bug（REGRESSION_ARCHIVE.md Case 227–244，均已归档）

参考模式 prealign：MP3 数字后缀收集、round-trip 计数、空 raw timeline、
shard 命名空间 skip、raw manifest 用 output、下游 expected_stems、postprocess 守恒、
authority English 越界、incomplete CTC 对齐、空参考词 unavailable、空/日语词分母不一致。

fallback 模式：句首 BREATHING/COUGH NVV 绑定、候选映射 skip、TextGrid/tokens 不匹配(RIA)、
summary 刷新、严格英文词典、序列化 locator 不唯一。

数据语义：`・`(U+30FB) 非日语、颜文字丢弃(has_japanese 拦平假名)、性别标签 `{M#..}{F#..}` 去标点留两版。

关键代码改动：
- `scripts/ctc_prealign.py`：`has_japanese`(Case242)、`strip_gender_tags`(Case244)、
  `_merge_reference_english_fragments`(Case227)、`attach_nvasr_candidate_provenance`
  reference_mode/句首 NVV drop(Case228/237)、`_validate_all_ctc_bundles` 丢无效 bundle(Case239/240)、
  `--stems-file` 过滤(Case238/241)
- `scripts/run_pipeline.py`：`_ctc_output_stems`(Case232)、`_seal_ctc_raw`(Case231)、
  `_refresh_postprocess_accounting`(Case233)、`_freeze_pre_ctc_stems` 空/日语排除(Case241)、
  `_load_ctc_accounting` 采纳更窄 eligible(Case241)
- `scripts/audit_strict_ok.py`：`expected` 用 `.lab` 集合(Case233)
- `scripts/stage_gamedata.py`：新增落地脚本

## 五、待继续（新会话要做的）

1. 监控后台任务 `b7wdvf60e`（`tail -F /tmp/claude-1000/.../tasks/b7wdvf60e.output | grep -E "START|DONE"`）。
2. 若某游戏 `DONE rc=1`：看 `/tmp/run_<codename>.log` 定位报错，按上面 Case 套路修，
   清 workspace `rm -rf /mnt/nvme3/mfa_work_gamedata_<codename>_20260903`，再单独重跑。
3. 全部 `rc=0` 后，确认 8 游戏都在 `/mnt/Raw/GAMEDATA_对齐_20260903/` 下。
4. 最后跑一次全量 pytest + `git diff --check` 收尾。

## 六、关键命令

```bash
cd /mnt/local_E/MFA_Pause/repo
# 参考模式单游戏
CUDA_VISIBLE_DEVICES=5,6 /home/user/miniconda3/envs/mfa-dev/bin/python scripts/run_pipeline.py \
  --config configs/gamedata_<codename>_reference_20260903.yaml \
  --python /home/user/miniconda3/envs/mfa-dev/bin/python
# 无参考模式单游戏（同 run_pipeline，非 streaming）
CUDA_VISIBLE_DEVICES=5,6 ... scripts/run_pipeline.py --config configs/gamedata_<codename>_noref_20260903.yaml ...
# 定向测试
PYTHONDONTWRITEBYTECODE=1 /home/user/miniconda3/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_ctc_english_units.py tests/test_nvasr_candidate_timeline.py tests/test_ctc_all_gpu_merge_receipts.py
```

## 七、注意事项

- 工作树本来就是脏的（见 `handoffs/20260824T103022Z-*.md`），本次改动是叠加的，禁止 reset/checkout/clean。
- GPU 被外部任务共享，只用 5,6 卡；外部任务间歇占用，显存会波动。
- MFA（CPU 对齐）是主要耗时：约 2~3 小时/万条；原神 113k 预计 ~24 小时。
- 无参考 fallback 过滤率高（ASR 噪声），白荆回廊 1421/5550 属正常。
- 需要看某个游戏已发布数量：`ls /mnt/Raw/GAMEDATA_对齐_20260903/<codename>/output` 或看 log 里的 `Postprocess contract`。
