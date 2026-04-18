#!/bin/bash
#SBATCH --job-name=marl_training
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=150GB
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email@university.edu

# ============================================================================
# SLURM Training Script for Light-MALib Multi-Agent Reinforcement Learning
# ============================================================================
# Resource Optimization for Northeastern Discovery Cluster:
#   GPU choices (pending jobs): v100-sxm2: 13 | h200: 25 | a100: ~5-10
#   Recommended: 1 GPU + 28 CPUs + 150GB (optimal for single-GPU training)
#   For faster scheduling of short jobs: Add --partition=gpu-short (max 2h)
#   To use H200 GPU: Add --gres=gpu:h200
#
# This script submits a training job using sbatch with support for:
#   - Fresh training runs
#   - Continued training from existing checkpoints
#   - Configurable resources via YAML config files
#   - Distributed Ray cluster setup
#
# Usage:
#   sbatch train.slurm --config <config_path> [--checkpoint <checkpoint_dir>]
#
# Examples:
#   # Fresh training
#   sbatch train.slurm --config expr_configs/cooperative_MARL_benchmark/academy/3_vs_1_with_keeper/ippo.yaml
#
#   # Continue training from checkpoint
#   sbatch train.slurm --config expr_configs/cooperative_MARL_benchmark/academy/3_vs_1_with_keeper/ippo.yaml \
#       --checkpoint logs/gr_football/benchmark_academy_3_vs_1_with_keeper_ippo/2024-01-15-10-30-45
# ============================================================================

set -e  # Exit on error
set -o pipefail  # python | tee must propagate failure (otherwise job shows SUCCESS on crash)

# Parse command-line arguments
CONFIG_PATH=""
CHECKPOINT_DIR=""
AUTO_RESUME="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --auto-resume)
            AUTO_RESUME="true"
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validation
if [ -z "$CONFIG_PATH" ]; then
    echo "Error: --config argument is required"
    echo "Usage: sbatch train.slurm --config <config_path> [--checkpoint <checkpoint_dir>]"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

if [ -n "$CHECKPOINT_DIR" ] && [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found at $CHECKPOINT_DIR"
    exit 1
fi

# Setup paths
# Use SLURM_SUBMIT_DIR which is the directory where sbatch was called
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-.}"
cd "$PROJECT_ROOT" || exit 1

# Create logs directory if it doesn't exist
mkdir -p logs

# Setup logging
LOG_FILE="logs/training_${SLURM_JOB_ID}.log"
{
    echo "=========================================="
    echo "Training Job Started"
    echo "=========================================="
    echo "Job ID: $SLURM_JOB_ID"
    echo "Node: $SLURMD_NODENAME"
    echo "GPUs: $SLURM_GPUS_ON_NODE"
    echo "Config: $CONFIG_PATH"
    if [ -n "$CHECKPOINT_DIR" ]; then
        echo "Resuming from: $CHECKPOINT_DIR"
    else
        echo "Fresh training run"
    fi
    echo "Start time: $(date)"
    echo "=========================================="
} | tee "$LOG_FILE"

# Conda: default name grf_env. Use full path if that env lives under another user, e.g.:
#   export CONDA_ENV_NAME=/home/chen.sihe1/.conda/envs/grf_env
#   python scripts/submit_training.py submit --config ...   # passes via --export when set
#   # or: sbatch --export=ALL,CONDA_ENV_NAME=$CONDA_ENV_NAME train.slurm --config ...
source ~/.bashrc
set +u
conda activate "${CONDA_ENV_NAME:-grf_env}"
set -u

# Shared conda envs often ship a .pth that adds another user's GRF_MARL to sys.path.
# Put this submission directory first so `import light_malib` always matches this checkout.
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# Load required modules (adjust for your HPC environment)
module load cuda/12.1.1
# module load cudnn/8.6.0
# module load python/3.10

echo "Python executable: $(which python)" | tee -a "$LOG_FILE"
echo "Python version: $(python --version)" | tee -a "$LOG_FILE"

# Ensure this job does not inherit a login-shell Ray address (we set our own below).
unset RAY_IP

# Per-job Ray temp dir so we never attach to another user's / stale /tmp/ray session
export RAY_TMPDIR="/tmp/ray_${USER:-user}_${SLURM_JOB_ID:-local}"
mkdir -p "$RAY_TMPDIR"
ray stop --force 2>/dev/null || true
# Do not rm /tmp/ray/session_latest: on shared nodes that breaks other users' default Ray sessions.
unset RAY_memory RAY_object_store_memory 2>/dev/null || true

# Only set CUDA_VISIBLE_DEVICES if not already set (e.g. by SLURM when multiple jobs share a node)
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

# Ray GCS port: do not use only SLURM_JOB_ID%N (IDs differing by N collide; port may be busy anyway).
# Default: ask OS for a free port on 127.0.0.1; retry ray start a few times. Override with RAY_HEAD_PORT.
unset RAY_ADDRESS

pick_free_ray_port() {
    python -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()"
}

export PYTHONUNBUFFERED=1

# Start Ray head in the background; driver connects via RAY_ADDRESS (see main_pbt.start_cluster).
RAY_NUM_CPU="${SLURM_CPUS_PER_TASK:-16}"
# Prefer GPUs visible to this step (SLURM_GPUS_ON_NODE can be the full node count).
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    RAY_NUM_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
else
    RAY_NUM_GPU="${SLURM_GPUS_ON_NODE:-1}"
    case "$RAY_NUM_GPU" in ''|*[!0-9]*) RAY_NUM_GPU=1 ;; esac
fi

RAY_START_OK=0
if [ -n "${RAY_HEAD_PORT:-}" ]; then
    export RAY_ADDRESS="127.0.0.1:${RAY_HEAD_PORT}"
    echo "Starting Ray head on fixed port ${RAY_HEAD_PORT} (RAY_ADDRESS=${RAY_ADDRESS}) temp=${RAY_TMPDIR}" | tee -a "$LOG_FILE"
    if ray start --head \
        --port="${RAY_HEAD_PORT}" \
        --temp-dir="${RAY_TMPDIR}" \
        --num-cpus="${RAY_NUM_CPU}" \
        --num-gpus="${RAY_NUM_GPU}" \
        --disable-usage-stats 2>&1 | tee -a "$LOG_FILE"; then
        RAY_START_OK=1
    fi
else
    for _ray_try in 1 2 3 4 5 6 7 8; do
        export RAY_HEAD_PORT="$(pick_free_ray_port)" || { echo "pick_free_ray_port failed" | tee -a "$LOG_FILE"; break; }
        export RAY_ADDRESS="127.0.0.1:${RAY_HEAD_PORT}"
        echo "Starting Ray head try ${_ray_try} on port ${RAY_HEAD_PORT} (RAY_ADDRESS=${RAY_ADDRESS}) temp=${RAY_TMPDIR}" | tee -a "$LOG_FILE"
        if ray start --head \
            --port="${RAY_HEAD_PORT}" \
            --temp-dir="${RAY_TMPDIR}" \
            --num-cpus="${RAY_NUM_CPU}" \
            --num-gpus="${RAY_NUM_GPU}" \
            --disable-usage-stats 2>&1 | tee -a "$LOG_FILE"; then
            RAY_START_OK=1
            break
        fi
        echo "ray start failed on port ${RAY_HEAD_PORT}; ray stop and retry..." | tee -a "$LOG_FILE"
        ray stop --force 2>/dev/null || true
    done
fi

if [ "$RAY_START_OK" != 1 ]; then
    echo "ray start --head failed after retries. Logs: $LOG_FILE and $RAY_TMPDIR" | tee -a "$LOG_FILE"
    exit 1
fi

echo "RAY_TMPDIR=$RAY_TMPDIR" | tee -a "$LOG_FILE"
echo "RAY_ADDRESS=$RAY_ADDRESS" | tee -a "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"
echo "SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE:-unset} MB" | tee -a "$LOG_FILE"

python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'; print(f'GPU OK: {torch.cuda.get_device_name(0)}')" \
    | tee -a "$LOG_FILE" || { echo "CUDA check failed." | tee -a "$LOG_FILE"; exit 1; }

# Dynamically find the latest checkpoint if auto-resume is enabled
if [ "$AUTO_RESUME" = "true" ] && [ -z "$CHECKPOINT_DIR" ]; then
    echo "Auto-resume enabled. Searching for latest checkpoint dynamically..." | tee -a "$LOG_FILE"
    
    LATEST_CHECKPOINT=$(python -c "
import yaml, os
from pathlib import Path
try:
    with open('$CONFIG_PATH', 'r') as f:
        cfg = yaml.safe_load(f)
    log_dir = cfg.get('log_dir', './logs').strip('./')
    if not log_dir: log_dir = 'logs'
    expr_group = cfg.get('expr_group', '')
    expr_name = cfg.get('expr_name', '')
    p = Path(log_dir) / expr_group / expr_name
    if p.exists():
        checkpoints = sorted([d for d in p.iterdir() if d.is_dir()])
        if checkpoints:
            print(str(checkpoints[-1]))
except Exception as e:
    pass
")
    
    if [ -n "$LATEST_CHECKPOINT" ]; then
        CHECKPOINT_DIR="$LATEST_CHECKPOINT"
        echo "Found latest checkpoint: $CHECKPOINT_DIR" | tee -a "$LOG_FILE"
    else
        echo "No checkpoints found. Will start fresh training." | tee -a "$LOG_FILE"
    fi
    echo "" | tee -a "$LOG_FILE"
fi

# Prepare training command
TRAIN_CMD="python light_malib/main_pbt.py --config $CONFIG_PATH"

if [ -n "$CHECKPOINT_DIR" ]; then
    # For checkpoint resumption, we need to modify the config dynamically
    # or pass it as an environment variable to be used by the training script
    export RESUME_FROM_CHECKPOINT="$CHECKPOINT_DIR"
    TRAIN_CMD="$TRAIN_CMD --resume-from $CHECKPOINT_DIR"
    echo "Resume mode enabled: $RESUME_FROM_CHECKPOINT" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "Running: $TRAIN_CMD" | tee -a "$LOG_FILE"
echo "=========================================="
echo "" | tee -a "$LOG_FILE"

# Run training
if eval "$TRAIN_CMD" 2>&1 | tee -a "$LOG_FILE"; then
    TRAIN_STATUS="SUCCESS"
    EXIT_CODE=0
else
    TRAIN_STATUS="FAILED"
    EXIT_CODE=$?
fi

# Final logging
{
    echo ""
    echo "=========================================="
    echo "Training Job Finished"
    echo "=========================================="
    echo "Status: $TRAIN_STATUS"
    echo "Exit Code: $EXIT_CODE"
    echo "End time: $(date)"
    echo "=========================================="
} | tee -a "$LOG_FILE"

# Move SLURM logs to experiment directory if training succeeded
if [ $EXIT_CODE -eq 0 ] && [ -n "$CHECKPOINT_DIR" ]; then
    SLURM_LOG_FILE="logs/slurm_${SLURM_JOB_ID}.log"
    if [ -f "$SLURM_LOG_FILE" ]; then
        cp "$SLURM_LOG_FILE" "$CHECKPOINT_DIR/slurm_resumed_${SLURM_JOB_ID}.log" 2>/dev/null || true
    fi
fi

# Tear down Ray head started above (ray.shutdown() in Python only disconnects the driver).
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER:-user}_${SLURM_JOB_ID:-local}}"
ray stop --force 2>/dev/null || true

exit $EXIT_CODE
