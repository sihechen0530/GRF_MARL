#!/usr/bin/env bash
set -euo pipefail

# Submit seed sweeps for the counterattack experiments.
#
# Usage:
#   bash scripts/submit_counterattack_seed_sweep.sh --dry-run
#   bash scripts/submit_counterattack_seed_sweep.sh --submit
#
# The script creates derived YAML configs under expr_configs/seed_sweeps/
# with unique expr_name values, then uses chain-submit so each seed can
# continue cleanly from its own checkpoints.

MODE="${1:---dry-run}"
if [[ "$MODE" != "--dry-run" && "$MODE" != "--submit" ]]; then
  echo "Usage: $0 [--dry-run|--submit]"
  exit 2
fi

NO_SUBMIT_FLAG="--no-submit"
if [[ "$MODE" == "--submit" ]]; then
  NO_SUBMIT_FLAG=""
fi

PYTHON_BIN="/home/chen.sihe1/.conda/envs/grf_env/bin/python"
CONFIG_DIR="expr_configs/seed_sweeps/counterattack"
mkdir -p "$CONFIG_DIR"

# Keep the default batch small enough for the cluster's submit limit.
# Start with MAPPO seed variance; add IPPO/MAT once MAPPO variance is established.
#
# Override examples:
#   SEEDS_LIST="271 503" bash scripts/submit_counterattack_seed_sweep.sh --submit
#   ALGORITHMS_LIST="ippo mappo mat" SEEDS_LIST="42" bash scripts/submit_counterattack_seed_sweep.sh --dry-run
read -r -a ALGORITHMS <<< "${ALGORITHMS_LIST:-mappo}"
read -r -a SEEDS <<< "${SEEDS_LIST:-42 137}"
ROLLOUT_SEED_MULT=7
NUM_JOBS="${NUM_JOBS:-3}"

for algo in "${ALGORITHMS[@]}"; do
  base="expr_configs/cooperative_MARL_benchmark/academy/counterattack/${algo}.yaml"
  for seed in "${SEEDS[@]}"; do
    rollout_seed=$((seed * ROLLOUT_SEED_MULT + 1))
    out="${CONFIG_DIR}/${algo}_seed${seed}.yaml"

    "$PYTHON_BIN" - "$base" "$out" "$algo" "$seed" "$rollout_seed" <<'PY'
import pathlib
import sys
import yaml

base, out, algo, seed, rollout_seed = sys.argv[1:]
seed = int(seed)
rollout_seed = int(rollout_seed)

with open(base) as f:
    cfg = yaml.safe_load(f)

cfg["seed"] = seed
cfg["rollout_manager"]["seed"] = rollout_seed
cfg["expr_name"] = f"seed_sweep_academy_counterattack_{algo}_seed{seed}"

out_path = pathlib.Path(out)
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
PY

    echo
    echo "# ${algo^^} counterattack seed=${seed} rollout_seed=${rollout_seed}"
    echo "python scripts/submit_training.py chain-submit --config ${out} --num-jobs ${NUM_JOBS} --job-name seed_${algo}_s${seed} --fresh --no-eval ${NO_SUBMIT_FLAG}"
    python scripts/submit_training.py chain-submit \
      --config "$out" \
      --num-jobs "$NUM_JOBS" \
      --job-name "seed_${algo}_s${seed}" \
      --fresh \
      --no-eval \
      ${NO_SUBMIT_FLAG}
  done
done

cat <<'EOF'

After training finishes, run evaluation manually, for example:

  /home/chen.sihe1/.conda/envs/grf_env/bin/python scripts/eval_checkpoint.py \
    --config expr_configs/seed_sweeps/counterattack/mappo_seed42.yaml \
    --run_dir logs/gr_football/seed_sweep_academy_counterattack_mappo_seed42/<timestamp> \
    --all-checkpoints \
    --num-games 100 \
    --output_dir results/seed_sweep_counterattack_mappo_seed42

EOF
