#!/bin/bash
#SBATCH --job-name=eureka_eval_revise
#SBATCH --output=logs/slurm_eval_%j.log
#SBATCH --error=logs/slurm_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH --partition=sharing

# ============================================================================
# Eval + Metrics + LLM Phi Revision (one step of the Iterative Eureka loop)
#
# This script is meant to be submitted as a SLURM job between training
# segments.  It:
#   1. Finds the latest checkpoint from the training config
#   2. Runs evaluation games to collect behavioral metrics
#   3. Calls the LLM to diagnose and revise the phi function
#   4. Writes the updated phi so the next training job picks it up
#
# Usage (standalone):
#   sbatch eval_revise.slurm --config <yaml> --iteration 1
#
# In an automated chain the iterative_eureka.py slurm-loop command handles
# all arguments and dependencies.
# ============================================================================

set -e
set -o pipefail

# --- Parse arguments --------------------------------------------------------
CONFIG_PATH=""
ITERATION=1
NUM_EVAL_GAMES=20
NUM_EVAL_WORKERS=4
PHI_DIR="generated_phi"
SCENARIO_NAME="GRF 11v11 full game"
HINT=""
TEMPERATURE=0.3

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)        CONFIG_PATH="$2";       shift 2 ;;
        --iteration)     ITERATION="$2";         shift 2 ;;
        --num-eval-games)  NUM_EVAL_GAMES="$2";  shift 2 ;;
        --num-eval-workers) NUM_EVAL_WORKERS="$2"; shift 2 ;;
        --phi-dir)       PHI_DIR="$2";           shift 2 ;;
        --scenario-name) SCENARIO_NAME="$2";     shift 2 ;;
        --hint)          HINT="$2";              shift 2 ;;
        --temperature)   TEMPERATURE="$2";       shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$CONFIG_PATH" ]; then
    echo "Error: --config is required"
    exit 1
fi

# --- Setup ------------------------------------------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-.}"
cd "$PROJECT_ROOT" || exit 1
mkdir -p logs metrics "$PHI_DIR"

source ~/.bashrc
set +u
conda activate "${CONDA_ENV_NAME:-grf_env}"
set -u

module load cuda/12.1.1

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
unset RAY_ADDRESS
unset RAY_IP

echo "=========================================="
echo "Iterative Eureka — Eval + Revise"
echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Iteration:  $ITERATION"
echo "Config:     $CONFIG_PATH"
echo "Eval games: $NUM_EVAL_GAMES"
echo "Phi dir:    $PHI_DIR"
echo "Start:      $(date)"
echo "=========================================="

# --- Step 0: Find latest checkpoint ----------------------------------------
CHECKPOINT=$(python -c "
import yaml
from pathlib import Path
with open('$CONFIG_PATH') as f:
    cfg = yaml.safe_load(f)
log_dir = cfg.get('log_dir', './logs').strip('./')
if not log_dir: log_dir = 'logs'
expr_group = cfg.get('expr_group', '')
expr_name  = cfg.get('expr_name', '')
p = Path(log_dir) / expr_group / expr_name
if p.exists():
    ckpts = sorted([d for d in p.iterdir() if d.is_dir()])
    if ckpts:
        print(str(ckpts[-1]))
")

if [ -z "$CHECKPOINT" ]; then
    echo "WARNING: No checkpoint found for config $CONFIG_PATH"
    echo "Previous training may have failed before saving any checkpoint. Skipping eval+revise."
    exit 0
fi
echo "Checkpoint: $CHECKPOINT"

# --- Step 1: Extract behavioral metrics ------------------------------------
METRICS_FILE="metrics/behavior_iter${ITERATION}.json"
echo ""
echo "[Step 1] Extracting behavioral metrics → $METRICS_FILE"

python scripts/extract_behavior_metrics.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG_PATH" \
    --num-games "$NUM_EVAL_GAMES" \
    --num-workers "$NUM_EVAL_WORKERS" \
    --output "$METRICS_FILE" \
    --phi-module "generated_phi.phi_llm"

echo "Metrics extracted successfully."

# --- Step 2: LLM diagnose + revise phi -------------------------------------
CURRENT_PHI="${PHI_DIR}/phi_llm.py"
NEW_PHI="${PHI_DIR}/phi_llm_v$((ITERATION + 1)).py"
echo ""
echo "[Step 2] LLM diagnosis + revision → $NEW_PHI"

REVISE_ARGS=(
    scripts/iterative_eureka.py revise
    --metrics "$METRICS_FILE"
    --current-phi "$CURRENT_PHI"
    --output "$NEW_PHI"
    --iteration "$ITERATION"
    --scenario-name "$SCENARIO_NAME"
    --temperature "$TEMPERATURE"
)
if [ -n "$HINT" ]; then
    REVISE_ARGS+=(--hint "$HINT")
fi

python "${REVISE_ARGS[@]}"

# --- Step 3: Activate the new phi ------------------------------------------
echo ""
echo "[Step 3] Activating new phi as ${PHI_DIR}/phi_llm.py"
cp "$NEW_PHI" "${PHI_DIR}/phi_llm.py"
echo "Done. Next training job will use the updated phi."

echo ""
echo "=========================================="
echo "Eval + Revise completed successfully"
echo "End: $(date)"
echo "=========================================="
