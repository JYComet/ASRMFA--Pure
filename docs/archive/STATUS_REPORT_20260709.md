# 合成ria_all Pipeline — Status Report

**Started:** 2026-07-09 17:14 UTC  
**Config:** `configs/合成ria_all.yaml`  
**Mode:** ctc_ready (NVASR CTC 已预跑完)  
**Files:** 100,940 audio stems  
**GPUs:** 8× NVIDIA RTX 4090 D (MFA num_jobs=8)

---

## What Was Done

### 1. Code Fixes (all in `scripts/run_pipeline.py`)
- **Cross-platform path translation**: Windows UNC paths (`\\RS3621\...`) auto-translate to Linux SMB mounts (`/mnt/Raw/...`). Works on both Windows and Linux.
- **Nested CTC directory support**: CTC files in nested subdirectories (`合成ria_00001/合成ria_00001.lab`) are now properly discovered — this was a critical bug.
- **In-place audio reference**: Audio files are NOT copied — the pipeline references them directly from the source directory to avoid copying 100k+ WAV files across SMB mounts.
- **Symlink preference**: Link strategy is symlink → hardlink → copy (avoids unnecessary copies).
- **Enhanced MFA Python detection**: Added `mfa-dev` and `asr` to conda env search list.

### 2. Environment Setup
- Using `/home/user/miniconda3/envs/mfa-dev/` (Python 3.11.15 + MFA 3.3.9)
- Installed missing package: `cn2an` (numeral normalization)
- All critical deps verified: montreal_forced_aligner, soundfile, pypinyin, yaml, numpy, scipy, librosa, praatio

### 3. Path Mapping
| Windows UNC | Linux Mount |
|---|---|
| `\\RS3621\Research_TTS\Data\Raw` | `/mnt/Raw/` |
| Audio source | `/mnt/Raw/v5_0707/合成ria/wavs` (100,940 WAVs) |
| CTC source | `/mnt/Raw/ASRNEW/合成ria/wavs` (nested, 100,940 dirs) |
| Final output | `/mnt/Raw/ASR_MFA/hechengria_all/` |

### 4. Pipeline Steps (ctc_ready mode)
1. **link** — Scan + link CTC files → workspace/ctc_pretg/  (IN PROGRESS)
2. **normalize_en** — Normalise English-word fragments in CTC output
3. **resample** — Resample audio to 16kHz → workspace/audio_16k/
4. **adjust** — Energy-based CTC boundary adjustment
5. **validate** — MFA corpus validation
6. **align** — MFA forced alignment (num_jobs=8)
7. **postprocess** — Final TextGrid post-processing → output/

---

## How to Check Progress

```bash
# Quick status
ws="/mnt/local_E/Audio Event Detection/chinese_mfa_pipeline/output/hechengria_all"
echo "CTC: $(find $ws/ctc_pretg -name '*.lab' | wc -l) / 100940"
echo "16k audio: $(find $ws/audio_16k -name '*.wav' | wc -l)"
echo "Aligned: $(find $ws/aligned -name '*.TextGrid' | wc -l)"
echo "Output: $(find /mnt/Raw/ASR_MFA/hechengria_all -name '*.TextGrid' | wc -l)"

# Is pipeline running?
ps aux | grep "run_pipeline.*合成ria_all" | grep -v grep

# View logs (output is unbuffered — use PYTHONUNBUFFERED=1 for future runs)
ls -la logs/

# Run monitor
bash monitor.sh
```

## Auto-Recovery

If the pipeline crashes, start the daemon:
```bash
nohup bash /mnt/local_E/Audio Event Detection/chinese_mfa_pipeline/daemon.sh &>/dev/null &
```

Or install the crontab for automatic checks every 5 minutes:
```bash
crontab -l 2>/dev/null > /tmp/ctab
cat /mnt/local_E/Audio Event Detection/chinese_mfa_pipeline/crontab_entry.txt >> /tmp/ctab
crontab /tmp/ctab
```

The pipeline is idempotent — on restart, completed files are skipped automatically.

---

## Output Location
Final post-processed TextGrids will be at:
`/mnt/Raw/ASR_MFA/hechengria_all/`  (= `\\RS3621\Research_TTS\Data\Raw\ASR_MFA\hechengria_all`)
