#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Multi-GPU Pipeline Launcher
# ═══════════════════════════════════════════════════════════════════════════
#
# Distributes the MFA pipeline across all available GPUs, each running
# independent streaming_pipeline.py processes with their own CUDA device.
#
# Usage:
#   # Auto-detect GPUs, use batch cache from config:
#   bash scripts/launch_multi_gpu.sh --config configs/batch_all.yaml
#
#   # Explicit GPU count, limit datasets per GPU:
#   bash scripts/launch_multi_gpu.sh --config configs/batch_all.yaml --gpus 4 --limit 10
#
#   # Use streaming mode (NAS→local SSD→NAS) with per-GPU log dir:
#   bash scripts/launch_multi_gpu.sh --config configs/batch_all.yaml \
#       --streaming --local-work /ssd/mfa_work --batch-size 2000
#
#   # Resume using existing checkpoint (skip completed datasets):
#   bash scripts/launch_multi_gpu.sh --config configs/batch_all.yaml --resume
#
#   # Dry-run: show what would be launched, don't execute:
#   bash scripts/launch_multi_gpu.sh --config configs/batch_all.yaml --dry-run
#
# Environment:
#   MFA_PYTHON     Path to Python with MFA installed (auto-detect if unset)
#   MFA_ROOT_DIR   Path to MFA models (auto-set if unset)
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──────────────────────────────────────────────────────────────
NUM_GPUS=""
CONFIG=""
DATASET_LIMIT=0        # 0 = all
BATCH_SIZE=""
STREAMING=false
LOCAL_WORK=""
DRY_RUN=false
RESUME=false
MFA_JOBS=""            # MFA num_jobs per worker (empty = auto)
PARALLEL_DATASETS=""   # streaming workers per GPU
LOG_DIR=""
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Colors ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

usage() {
    cat <<EOF
Usage: $0 --config <yaml> [OPTIONS]

Options:
  --config PATH         Pipeline config file (required)
  --gpus N              Number of GPUs to use (default: auto-detect)
  --limit N             Limit datasets per GPU (0=all)
  --batch-size N        Stems per batch (streaming mode)
  --streaming           Use streaming_pipeline.py (NAS→local SSD→NAS)
  --local-work DIR      Local SSD path for streaming (can be comma-separated)
  --parallel N          Streaming workers per GPU (default: from config)
  --mfa-jobs N          MFA num_jobs per worker (default: auto)
  --log-dir DIR         Log output directory (default: logs/multi_gpu_<ts>)
  --resume              Skip completed datasets (uses checkpoint)
  --dry-run             Print launch plan without executing
  -h, --help            Show this help
EOF
    exit 0
}

# ── Parse args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG="$2"; shift 2 ;;
        --gpus)        NUM_GPUS="$2"; shift 2 ;;
        --limit)       DATASET_LIMIT="$2"; shift 2 ;;
        --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
        --streaming)   STREAMING=true; shift ;;
        --local-work)  LOCAL_WORK="$2"; shift 2 ;;
        --parallel)    PARALLEL_DATASETS="$2"; shift 2 ;;
        --mfa-jobs)    MFA_JOBS="$2"; shift 2 ;;
        --log-dir)     LOG_DIR="$2"; shift 2 ;;
        --resume)      RESUME=true; shift ;;
        --dry-run)     DRY_RUN=true; shift ;;
        -h|--help)     usage ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; usage ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo -e "${RED}ERROR: --config is required${NC}"
    usage
fi

# ── Auto-detect GPUs ──────────────────────────────────────────────────────
if [[ -z "$NUM_GPUS" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    elif command -v python3 &>/dev/null; then
        NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    fi
    NUM_GPUS=${NUM_GPUS:-1}
    if [[ "$NUM_GPUS" -lt 1 ]]; then NUM_GPUS=1; fi
fi

# ── Resolve paths ──────────────────────────────────────────────────────────
CONFIG_PATH="$CONFIG"
[[ "$CONFIG_PATH" = /* ]] || CONFIG_PATH="$PROJECT_ROOT/$CONFIG"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo -e "${RED}ERROR: Config not found: $CONFIG_PATH${NC}"
    exit 1
fi

# Determine Python with MFA
MFA_PYTHON="${MFA_PYTHON:-}"
if [[ -z "$MFA_PYTHON" ]]; then
    # Auto-detect: prefer mfa-dev env
    for py in \
        "$HOME/miniconda3/envs/mfa-dev/bin/python3" \
        "$HOME/miniconda3/envs/mfa-dev/bin/python" \
        "$HOME/miniconda3/envs/mfa_chinese/bin/python3" \
        "$HOME/miniconda3/envs/mfa_chinese/bin/python" \
        "$(which python3 2>/dev/null || true)" \
        "$(which python 2>/dev/null || true)"; do
        if [[ -x "$py" ]]; then
            MFA_PYTHON="$py"
            break
        fi
    done
fi

if [[ -z "$MFA_PYTHON" ]]; then
    echo -e "${RED}ERROR: Cannot find Python with MFA. Set MFA_PYTHON env var.${NC}"
    exit 1
fi
echo -e "${GREEN}MFA Python: $MFA_PYTHON${NC}"

# ── Resolve batch cache ────────────────────────────────────────────────────
CACHE_FILE="$PROJECT_ROOT/cache/$(basename "$CONFIG_PATH" .yaml).cache.json"
CKPT_FILE="$PROJECT_ROOT/cache/$(basename "$CONFIG_PATH" .yaml).checkpoint.json"

# ── Discover datasets ──────────────────────────────────────────────────────
if [[ -f "$CACHE_FILE" ]]; then
    n_total=$(python3 -c "import json; d=json.load(open('$CACHE_FILE')); print(len(d.get('datasets',[])))" 2>/dev/null || echo "0")
else
    echo -e "${YELLOW}Cache not found: $CACHE_FILE${NC}"
    echo -e "  Run scan first: python scripts/run_pipeline.py --config $CONFIG --scan-only"
    exit 1
fi

# ── Resume: skip completed datasets ────────────────────────────────────────
COMPLETED=0
if $RESUME && [[ -f "$CKPT_FILE" ]]; then
    COMPLETED=$(python3 -c "import json; d=json.load(open('$CKPT_FILE')); print(len(d.get('completed',[])))" 2>/dev/null || echo "0")
    echo -e "${GREEN}Resume: $COMPLETED completed, skipping${NC}"
fi

REMAINING=$((n_total - COMPLETED))
if [[ "$DATASET_LIMIT" -gt 0 ]] && [[ "$DATASET_LIMIT" -lt "$REMAINING" ]]; then
    REMAINING=$DATASET_LIMIT
fi

# Per-GPU dataset count (ceiling division)
PER_GPU=$(( (REMAINING + NUM_GPUS - 1) / NUM_GPUS ))
echo -e "${CYAN}Datasets: $n_total total, $COMPLETED completed → $REMAINING active${NC}"
echo -e "${CYAN}GPUs: $NUM_GPUS → ~$PER_GPU datasets per GPU${NC}"

if [[ "$REMAINING" -le 0 ]]; then
    echo -e "${GREEN}All datasets complete — nothing to do.${NC}"
    exit 0
fi

# ── Log directory ──────────────────────────────────────────────────────────
if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="$PROJECT_ROOT/logs/multi_gpu_$TIMESTAMP"
fi
mkdir -p "$LOG_DIR"

# ── Build per-GPU command ──────────────────────────────────────────────────
launch_gpu() {
    local gpu_id=$1
    local gpu_offset=$2
    local gpu_limit=$3
    local log_file="$LOG_DIR/gpu${gpu_id}.log"
    local pid_file="$LOG_DIR/gpu${gpu_id}.pid"
    local err_file="$LOG_DIR/gpu${gpu_id}.err"

    # Build streaming or direct command
    if $STREAMING; then
        cmd=(
            "$MFA_PYTHON" "$SCRIPT_DIR/streaming_pipeline.py"
            "--config" "$CONFIG_PATH"
            "--batch-cache" "$CACHE_FILE"
            "--gpus" "1"               # This process only sees 1 GPU
            "--parallel-datasets" "${PARALLEL_DATASETS:-1}"
            "--local-work" "$LOCAL_WORK"
        )
        [[ -n "$BATCH_SIZE" ]] && cmd+=("--batch-size" "$BATCH_SIZE")
        [[ "$gpu_limit" -gt 0 ]] && cmd+=("--limit-datasets" "$gpu_limit")
    else
        cmd=(
            "$MFA_PYTHON" "$SCRIPT_DIR/run_pipeline.py"
            "--config" "$CONFIG_PATH"
            "--mode" "batch_ctc_ready"
            "--device" "cuda:0"         # CUDA_VISIBLE_DEVICES remaps this to gpu_id
            "--dataset-offset" "$gpu_offset"
            "--dataset-limit" "$gpu_limit"
        )
    fi

    [[ -n "$MFA_JOBS" ]] && cmd+=("--mfa-jobs" "$MFA_JOBS")
    $RESUME || cmd+=("--no-resume")

    if $DRY_RUN; then
        echo "  GPU $gpu_id: CUDA_VISIBLE_DEVICES=$gpu_id"
        echo "    ${cmd[@]}"
        echo "    > $log_file 2> $err_file"
        return
    fi

    echo -e "${YELLOW}[GPU $gpu_id] Starting (log: $log_file)${NC}"
    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        export MFA_ROOT_DIR="${MFA_ROOT_DIR:-$PROJECT_ROOT/models/mfa}"
        echo "PID: $$" > "$pid_file"
        echo "Started: $(date -Iseconds)" >> "$pid_file"
        echo "Command: ${cmd[@]}" >> "$pid_file"
        exec "${cmd[@]}" >> "$log_file" 2>> "$err_file"
    ) &
    local pid=$!
    echo "$pid" > "$pid_file"
    echo -e "${GREEN}[GPU $gpu_id] PID=$pid${NC}"
}

# ═══════════════════════════════════════════════════════════════════════════
# Launch
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Launch Plan${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
echo "  Config:       $CONFIG_PATH"
echo "  Cache:        $CACHE_FILE"
echo "  GPUs:         $NUM_GPUS"
echo "  Datasets:     $n_total total, $REMAINING remaining"
echo "  Per GPU:      ~$PER_GPU"
echo "  Streaming:    $STREAMING"
[[ -n "$LOCAL_WORK" ]] && echo "  Local work:   $LOCAL_WORK"
echo "  Log dir:      $LOG_DIR"
echo "  Resume:       $RESUME"
echo ""

if $STREAMING; then
    # Streaming mode: all GPUs share the full dataset pool
    # streaming_pipeline.py handles internal GPU assignment via --gpus
    SINGLE_CMD=(
        "$MFA_PYTHON" "$SCRIPT_DIR/streaming_pipeline.py"
        "--config" "$CONFIG_PATH"
        "--batch-cache" "$CACHE_FILE"
        "--gpus" "$NUM_GPUS"
        "--parallel-datasets" "${PARALLEL_DATASETS:-$NUM_GPUS}"
    )
    [[ -n "$BATCH_SIZE" ]] && SINGLE_CMD+=("--batch-size" "$BATCH_SIZE")
    [[ -n "$LOCAL_WORK" ]] && SINGLE_CMD+=("--local-work" "$LOCAL_WORK")
    [[ "$DATASET_LIMIT" -gt 0 ]] && SINGLE_CMD+=("--limit-datasets" "$DATASET_LIMIT")
    [[ -n "$MFA_JOBS" ]] && SINGLE_CMD+=("--mfa-jobs" "$MFA_JOBS")
    $RESUME || SINGLE_CMD+=("--no-resume")

    if $DRY_RUN; then
        echo -e "${YELLOW}[DRY-RUN] Would execute:${NC}"
        echo "  ${SINGLE_CMD[@]}"
        echo ""
        echo "  (streaming_pipeline.py manages GPU assignment internally with --gpus $NUM_GPUS)"
    else
        echo -e "${YELLOW}[MAIN] Launching streaming pipeline ($NUM_GPUS GPUs, ${PARALLEL_DATASETS:-$NUM_GPUS} workers)${NC}"
        echo -e "${YELLOW}[MAIN] Log: $LOG_DIR/main.log${NC}"
        export CUDA_VISIBLE_DEVICES=""
        export MFA_ROOT_DIR="${MFA_ROOT_DIR:-$PROJECT_ROOT/models/mfa}"
        echo "PID: $$" > "$LOG_DIR/main.pid"
        echo "Started: $(date -Iseconds)" > "$LOG_DIR/main.pid"
        exec "${SINGLE_CMD[@]}" >> "$LOG_DIR/main.log" 2>> "$LOG_DIR/main.err"
    fi
else
    # Direct mode: launch 1 process per GPU, each with CUDA_VISIBLE_DEVICES=<gpu_id>
    # Each process gets ~1/N of the datasets
    for ((gpu=0; gpu < NUM_GPUS; gpu++)); do
        offset=$((gpu * PER_GPU))
        # Last GPU gets remaining
        if [[ $gpu -eq $((NUM_GPUS - 1)) ]]; then
            limit=$((REMAINING - gpu * PER_GPU))
        else
            limit=$PER_GPU
        fi
        launch_gpu "$gpu" "$offset" "$limit"
    done

    if ! $DRY_RUN; then
        echo ""
        echo -e "${GREEN}All $NUM_GPUS GPU processes launched.${NC}"
        echo "  Logs: $LOG_DIR/"
        echo "  PIDs: $(cat "$LOG_DIR"/gpu*.pid 2>/dev/null | head -$NUM_GPUS | tr '\n' ' ')"
        echo ""
        echo "  Monitor:  tail -f $LOG_DIR/gpu*.log"
        echo "  Status:   grep -E 'DONE|FAIL|ERROR' $LOG_DIR/gpu*.log"
        echo "  Kill all: kill $(cat "$LOG_DIR"/gpu*.pid 2>/dev/null | tr '\n' ' ')"
        echo ""
        echo "  Waiting for all processes to complete..."
        echo ""

        # Wait for all backgrounded GPUs
        wait

        # Summary
        echo ""
        echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
        echo -e "${CYAN}  Run Complete — Summary${NC}"
        echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
        for ((gpu=0; gpu < NUM_GPUS; gpu++)); do
            log="$LOG_DIR/gpu${gpu}.log"
            if [[ -f "$log" ]]; then
                done_count=$(grep -c "DONE" "$log" 2>/dev/null || echo "0")
                fail_count=$(grep -c "FAIL" "$log" 2>/dev/null || echo "0")
                echo -e "  GPU $gpu: ${GREEN}$done_count DONE${NC}, ${RED}$fail_count FAIL${NC}"
            fi
        done

        if [[ -f "$CKPT_FILE" ]]; then
            TOTAL_DONE=$(python3 -c "import json; d=json.load(open('$CKPT_FILE')); print(len(d.get('completed',[])))" 2>/dev/null || echo "?")
            TOTAL_FAIL=$(python3 -c "import json; d=json.load(open('$CKPT_FILE')); print(len(d.get('failed',[])))" 2>/dev/null || echo "?")
            echo "  ─────────────────────────────"
            echo "  Total: ${GREEN}$TOTAL_DONE completed${NC}, ${RED}$TOTAL_FAIL failed${NC}"
        fi
    fi
fi
