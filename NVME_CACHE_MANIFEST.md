# NVMe 音频缓存路径文档

> 自动生成于: 2026-08-04  
> 缓存脚本: `scripts/cache_audio_to_nvme.py`

## 缓存根目录

```
/mnt/nvme3/mfa_audio_cache/
```

## 目录结构

```
/mnt/nvme3/mfa_audio_cache/
├── cache_manifest.json          # 缓存元数据
├── ria/                         # 主播: ria (18,000 WAV, ~16 GB)
│   ├── 036000_弹幕互动_回应吐槽弹幕.wav
│   ├── 036001_弹幕互动_回应吐槽弹幕.wav
│   └── ...
├── 花礼/                        # 主播: 花礼 (18,000 WAV, ~16 GB)
│   ├── 018000_杂谈互动_日常分享.wav
│   ├── 018001_杂谈互动_日常分享.wav
│   └── ...
└── 雪狐桑/                      # 主播: 雪狐桑 (18,000 WAV, ~16 GB)
    ├── 000000_直播流程_开场介绍.wav
    ├── 000001_直播流程_开场介绍.wav
    └── ...
```

## 源数据位置 (NAS)

| 路径 | 说明 |
|------|------|
| `/mnt/Raw/新版合成英文数据/ria/` | ria 主播音频 (CIFS NAS) |
| `/mnt/Raw/新版合成英文数据/花礼/` | 花礼 主播音频 (CIFS NAS) |
| `/mnt/Raw/新版合成英文数据/雪狐桑/` | 雪狐桑 主播音频 (CIFS NAS) |

## 容量

| 项目 | 数值 |
|------|------|
| 总文件数 | 54,000 WAV |
| 总大小 | ~48 GB |
| NVMe 盘 | `/dev/nvme3n1` (7.0 TB, 6.6 TB 可用) |
| NVMe 写入速度 | ~1.3 GB/s |
| 全量复制耗时 | ~37 秒 |

## 使用方式

### 1. 首次初始化缓存

```bash
python scripts/cache_audio_to_nvme.py \
  --source /mnt/Raw/新版合成英文数据
```

### 2. 查看缓存状态

```bash
python scripts/cache_audio_to_nvme.py --status
```

### 3. 删除缓存

```bash
python scripts/cache_audio_to_nvme.py --remove
```

### 4. 主管线自动检测

`run_pipeline.py` 启动时自动检查 `/mnt/nvme3/mfa_audio_cache/`：

```
如果 NVMe 缓存存在且匹配源数据路径:
  → 使用永久缓存 (不自动删除)

如果 NVMe 缓存不存在:
  → 自动创建临时缓存到 /tmp/mfa_audio_cache/
  → 任务完成后自动清理
```

行为可通过 `--nvme-cache PATH` 手动指定缓存路径。

### 5. ctc_prealign.py 直接调用

```bash
# 先确保缓存存在
python scripts/cache_audio_to_nvme.py --status

# 指定 NVMe 缓存为音频源
python scripts/ctc_prealign.py \
  --data-dir /mnt/Raw/新版合成英文数据 \
  --audio-dir /mnt/nvme3/mfa_audio_cache \
  --all-gpus --overwrite \
  ...
```

## manifest 格式 (`cache_manifest.json`)

```json
{
  "version": 1,
  "created": "2026-08-04T12:00:00",
  "source": "/mnt/Raw/新版合成英文数据",
  "cache_root": "/mnt/nvme3/mfa_audio_cache",
  "total_files": 54000,
  "total_size_gb": 48.0,
  "speakers": {
    "ria":     {"files": 18000, "size_bytes": 17179869184},
    "花礼":    {"files": 18000, "size_bytes": 17179869184},
    "雪狐桑":  {"files": 18000, "size_bytes": 17179869184}
  }
}
```

## 相关文件

| 文件 | 说明 |
|------|------|
| [cache_audio_to_nvme.py](scripts/cache_audio_to_nvme.py) | 缓存管理脚本 |
| [run_pipeline.py](scripts/run_pipeline.py) | 主管线 (自动检测缓存) |
| [ctc_prealign.py](scripts/ctc_prealign.py) | CTC 预对齐 (通过 `--audio-dir` 使用) |
