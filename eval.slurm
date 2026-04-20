#!/bin/bash
#SBATCH --job-name=marl_eval
#SBATCH --output=logs/slurm_eval_%j.log
#SBATCH --error=logs/slurm_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128GB
#SBATCH --time=4:00:00
#SBATCH --partition=courses
#SBATCH --exclude=d3091

# ============================================================================
# SLURM Evaluation Script for Light-MALib
# Evaluates all checkpoints in a run directory against the built-in AI.
# Typically submitted as a dependent job after train.slurm via submit_training.py.
#
# Usage:
#   sbatch eval.slurm --config <config_path> [--run-dir <run_dir>]
#
# If --run-dir is omitted, the latest run directory for the experiment is used.
# ============================================================================

set -e

CONFIG_PATH=""
RUN_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CONFIG_PATH" ]; then
    echo "Error: --config argument is required"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-.}"
cd "$PROJECT_ROOT" || exit 1

# Parse experiment metadata from config
EXPR_NAME=$(grep '^expr_name:' "$CONFIG_PATH" | awk '{print $2}')
EXPR_GROUP=$(grep '^expr_group:' "$CONFIG_PATH" | awk '{print $2}')
EXPR_GROUP="${EXPR_GROUP:-gr_football}"

# Find run dir if not provided
if [ -z "$RUN_DIR" ]; then
    EXPR_BASE="$PROJECT_ROOT/logs/$EXPR_GROUP/$EXPR_NAME"
    RUN_DIR=$(ls -1dt "$EXPR_BASE"/????-??-??-??-??-?? 2>/dev/null | head -1)
    if [ -z "$RUN_DIR" ]; then
        echo "Error: No run directory found under $EXPR_BASE"
        exit 1
    fi
fi

if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Run directory not found: $RUN_DIR"
    exit 1
fi

# Redirect all output to the experiment directory alongside training logs
exec > "$RUN_DIR/slurm_eval_${SLURM_JOB_ID}.log" 2>&1

echo "=========================================="
echo "Evaluation Job Started"
echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "CPUs:       $SLURM_CPUS_ON_NODE"
echo "Config:     $CONFIG_PATH"
echo "Run dir:    $RUN_DIR"
echo "Start time: $(date)"
echo "=========================================="

source ~/.bashrc
set +u
conda activate /home/chen.sihe1/.conda/envs/grf_env
set -u

module load cuda/12.1.1

echo "Python executable: $(which python)"
echo ""

# Use 60 of the 64 CPUs for workers, leaving headroom for the main process
python scripts/eval_checkpoint.py \
    --config "$CONFIG_PATH" \
    --run_dir "$RUN_DIR" \
    --num_workers 60

echo ""
echo "=========================================="
echo "Evaluation Job Finished"
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
