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

# Activate conda environment (if using conda)
# Uncomment and modify if using conda
source ~/.bashrc
set +u
conda activate /home/chen.sihe1/.conda/envs/grf_env
set -u

# Load required modules (adjust for your HPC environment)
module load cuda/12.1.1
# module load cudnn/8.6.0
# module load python/3.10

echo "Python executable: $(which python)" | tee -a "$LOG_FILE"
echo "Python version: $(python --version)" | tee -a "$LOG_FILE"

# Ensure this job starts its own Ray cluster (do not connect to an existing one)
# Otherwise the training script would attach to a previous job's cluster with no free resources
unset RAY_ADDRESS
unset RAY_IP

# Set Ray configuration for SLURM environment
export RAY_memory=$((${SLURM_MEM_PER_NODE:-256} * 1024 * 1024 * 1024))  # Convert GB to bytes
export RAY_object_store_memory=$((${SLURM_MEM_PER_NODE:-256} * 1024 * 1024 * 1024 / 2))  # Half of total
# Only set CUDA_VISIBLE_DEVICES if not already set (e.g. by SLURM when multiple jobs share a node)
# so each job only sees its allocated GPU(s) and does not clash with others
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

# Optional: Configure logging
export PYTHONUNBUFFERED=1

echo "Ray Memory: $RAY_memory bytes" | tee -a "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" | tee -a "$LOG_FILE"

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

exit $EXIT_CODE
