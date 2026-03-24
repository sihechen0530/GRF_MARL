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
# Usage:
#   sbatch train.slurm --config <config_path>
#   sbatch train.slurm --config <config_path> --resume          # resume from latest run
#   sbatch train.slurm --config <config_path> --resume <dir>    # resume from specific dir
# ============================================================================

set -e  # Exit on error

# Parse command-line arguments
CONFIG_PATH=""
RESUME_DIR=""      # explicit dir to resume from (optional)
AUTO_RESUME="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --resume)
            AUTO_RESUME="true"
            # Optional: explicit directory after --resume
            if [[ -n "${2:-}" && "${2}" != --* ]]; then
                RESUME_DIR="$2"
                shift 2
            else
                shift 1
            fi
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
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

if [ -n "$RESUME_DIR" ] && [ ! -d "$RESUME_DIR" ]; then
    echo "Error: Resume directory not found at $RESUME_DIR"
    exit 1
fi

# Setup paths
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-.}"
cd "$PROJECT_ROOT" || exit 1

# Parse experiment metadata from config (no conda needed, just grep)
EXPR_NAME=$(grep '^expr_name:' "$CONFIG_PATH" | awk '{print $2}')
EXPR_GROUP=$(grep '^expr_group:' "$CONFIG_PATH" | awk '{print $2}')
EXPR_GROUP="${EXPR_GROUP:-gr_football}"
EXPR_BASE="$PROJECT_ROOT/logs/$EXPR_GROUP/$EXPR_NAME"

# Determine experiment output directory
if [ -n "$RESUME_DIR" ]; then
    # Explicit resume directory provided
    EXPR_LOG_DIR="$(cd "$RESUME_DIR" && pwd)"
elif [ "$AUTO_RESUME" = "true" ]; then
    # Find latest existing run dir; fall back to fresh if none exists
    LATEST=$(ls -1dt "$EXPR_BASE"/????-??-??-??-??-?? 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        EXPR_LOG_DIR="$LATEST"
    else
        EXPR_LOG_DIR="$EXPR_BASE/$(date +%Y-%m-%d-%H-%M-%S)"
    fi
else
    # Fresh run
    EXPR_LOG_DIR="$EXPR_BASE/$(date +%Y-%m-%d-%H-%M-%S)"
fi

mkdir -p "$EXPR_LOG_DIR"

# Redirect all subsequent output (stdout + stderr) to the experiment directory
exec > "$EXPR_LOG_DIR/slurm_${SLURM_JOB_ID}.log" 2>&1

echo "=========================================="
echo "Training Job Started"
echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPUs:       $SLURM_GPUS_ON_NODE"
echo "Config:     $CONFIG_PATH"
echo "Output dir: $EXPR_LOG_DIR"
if [ -d "$EXPR_LOG_DIR" ] && [ "$(ls -A "$EXPR_LOG_DIR" 2>/dev/null | grep -v "slurm_")" ]; then
    echo "Mode:       resume"
else
    echo "Mode:       fresh"
fi
echo "Start time: $(date)"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
set +u
conda activate /home/chen.sihe1/.conda/envs/grf_env
set -u

# Load required modules
module load cuda/12.1.1

echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

# Kill any stale Ray processes from previous jobs on this node, then start fresh
ray stop --force 2>/dev/null || true
unset RAY_ADDRESS
unset RAY_IP

# Set Ray configuration for SLURM environment
export RAY_memory=$((${SLURM_MEM_PER_NODE:-256} * 1024 * 1024 * 1024))
export RAY_object_store_memory=$((${SLURM_MEM_PER_NODE:-256} * 1024 * 1024 * 1024 / 2))
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
export PYTHONUNBUFFERED=1

echo "Ray Memory: $RAY_memory bytes"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# Run training — main_pbt.py resumes if EXPR_LOG_DIR already exists, starts fresh otherwise
TRAIN_CMD="python light_malib/main_pbt.py --config $CONFIG_PATH --expr-log-dir $EXPR_LOG_DIR"

echo "Running: $TRAIN_CMD"
echo "=========================================="
echo ""

if eval "$TRAIN_CMD"; then
    TRAIN_STATUS="SUCCESS"
    EXIT_CODE=0
else
    TRAIN_STATUS="FAILED"
    EXIT_CODE=$?
fi

echo ""
echo "=========================================="
echo "Training Job Finished"
echo "=========================================="
echo "Status:    $TRAIN_STATUS"
echo "Exit Code: $EXIT_CODE"
echo "End time:  $(date)"
echo "=========================================="

exit $EXIT_CODE
