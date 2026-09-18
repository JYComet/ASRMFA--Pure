# GAMEDATA 四游戏增量重建交接（2026-09-09）

## 最终状态

运行 `20260909T105050Z-fresh4` 已完成并发布。只重建了当前源数据中的
`reverse1999`、`persona`、`punishing_gray_raven`、`wuwa_new`；七个旧游戏
`snowbreak/huanxing/baijing/zhongmodi/yihuan/zzz/genshin` 的发布前后元数据
指纹完全一致。

源数据固定为只读 `/mnt/Raw/GAMEDATA`。同 basename、同目录且非空的 TXT
直接作为参考文本；没有有效 TXT 的条目进入 ASR fallback。所有源音频先转换为
单声道 PCM16 WAV，再执行 CTC/MFA/后处理。发布成品按
`<game>/<speaker>/<stem>` 分类，GAMESL 音频的首尾静音归一到 0.5 秒，TextGrid
同步映射到补静音后的完整时间轴。

| 游戏 | 源条目 | CTC 通过 | 最终通过 | 最终过滤 | 补静音后小时 | 说话人数 |
|---|---:|---:|---:|---:|---:|---:|
| reverse1999 | 25,513 | 25,138 | 13,578 | 11,935 | 26.601931 | 145 |
| persona | 26,742 | 26,652 | 3,903 | 22,839 | 4.918347 | 96 |
| punishing_gray_raven | 15,399 | 15,244 | 9,844 | 5,555 | 19.185031 | 42 |
| wuwa_new | 15,113 | 14,974 | 11,269 | 3,844 | 21.299652 | 109 |
| **合计** | **82,767** | **82,008** | **38,594** | **44,173** | **72.004961** | — |

过滤属于正式结果，不因正常过滤率高而重跑。`persona` 在最终验收时另发现 1 条
0.5 秒、peak/RMS 均为 0 的纯静音 WAV，已从发布集合剔除并可恢复地隔离；占该
游戏源条目的约 0.004%，占其原 3,904 条最终候选的约 0.026%。

## 最终公开结构

任务配置：`configs/gamedata_rebuild_20260909.yaml`。

```text
/mnt/Raw/GAMEDATA_对齐_20260903/
  reverse1999/<speaker>/<stem>.TextGrid
  persona/<speaker>/<stem>.TextGrid
  punishing_gray_raven/<speaker>/<stem>.TextGrid
  wuwa_new/<speaker>/<stem>.TextGrid

/mnt/Raw/GAMESL/
  reverse1999/<speaker>/<stem>.wav
  persona/<speaker>/<stem>.wav
  punishing_gray_raven/<speaker>/<stem>.wav
  wuwa_new/<speaker>/<stem>.wav
```

每个通过条目严格对应一对相同相对路径的 TextGrid/WAV。TextGrid tier 顺序固定为
`raw_text, pinyin, hanzi, words, pinyin_phones`；WAV 固定为 WAV 容器、单声道、
PCM16。四个游戏的 `default` 未解析说话人计数均为 0；`persona/NS未知说话人`
是源数据中明确存在的说话人目录，不是 fallback `default`。

## 验收证据

- NVMe 运行根：
  `/mnt/nvme3/gamedata_rebuild_20260909/runs/20260909T105050Z-fresh4`
- NAS 批准凭据：
  `/mnt/Raw/.gamedata_publish_staging/20260909T105050Z-fresh4/approvals`
- 公开路径逐条验收：`<run_root>/verification_public_v1`
- 发布前旧游戏基线：`<run_root>/gate5_prepublish_immutable_baseline_v3.json`
- 发布后旧游戏复核：`<run_root>/gate6_postpublish_immutable_check_v1.json`
- 发布事务日志：
  `/mnt/Raw/.gamedata_rebuild_archive/20260909T105050Z-fresh4/publish_journal.jsonl`
- 旧 reverse1999 可回滚归档：
  `/mnt/Raw/.gamedata_rebuild_archive/20260909T105050Z-fresh4/reverse1999/{aligned,gamesl}`

NVMe、NAS 私有 staging、最终公开路径三次验收的四个 per-game pair digest 完全
一致。验收逐条解析 TextGrid、检查五个 tier 和所有区间/全局时间轴，解码 WAV，
检查 WAV/PCM16/mono、有效语音及 0.5 秒边缘静音，并对 TextGrid/WAV 内容计算
SHA-256 绑定摘要。

## 本轮关键修复

1. 冷恢复 postprocess 会把 MFA `invalid` 条目遗漏：恢复范围现在合并当前精确的
   `missing + invalid`，并拒绝额外 stem。
2. native-anchor 校验对约 155MB 映射文件逐 TextGrid 重读重哈希：改为按 stat key
   缓存，消除 PB 级重复 I/O。
3. `--skip-to align` 冷恢复误把 CTC 已过滤条目重新放回执行集合：现在从密封 CTC
   receipt 恢复可执行 stem，同时保留正式分母。
4. 说话人最终化器错误优先读取原始容器：现在优先使用已验证的 staged PCM16 WAV。
5. 补静音后 TextGrid 只平移、不重建完整时间轴：现在所有 tier 都映射并覆盖
   `[0, padded_wav_duration]`，首尾/间隙补空 interval。
6. 固定 1024 样本静音检测存在帧相位漂移：最终化改用 10ms 滑窗 RMS，确保不同
   采样率下首尾 0.5 秒边界稳定；全量验收使用同一平移不变定义。

错误的第一版最终化输出和中断 rsync 的 12 个点号临时文件均在私有 staging 的
隔离目录内，没有发布，也没有删除正式源或成品。

最终代码验收命令使用 `/home/user/miniconda3/bin/python -m pytest`，覆盖 finalizer、
独立 staging verifier、增量 rebuild、冷恢复分母和 MFA runtime capability，共
`92 passed`；随后 `git diff --check` 退出码为 0。
