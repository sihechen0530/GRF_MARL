#!/usr/bin/env python3
"""
Evaluate saved checkpoints against the built-in AI.

Scans a run directory for all saved checkpoints (epoch_*, best, last),
runs each against the built-in opponent, and reports win rate with
confidence intervals.

Usage:
    # Evaluate all checkpoints from a specific run
    python scripts/eval_checkpoint.py \
        --config expr_configs/cooperative_MARL_benchmark/academy/3_vs_1_with_keeper/mappo.yaml \
        --run_dir logs/gr_football/benchmark_academy_3_vs_1_with_keeper_mappo/2026-03-14-12-00-00

    # Evaluate a single checkpoint
    python scripts/eval_checkpoint.py \
        --config expr_configs/cooperative_MARL_benchmark/academy/3_vs_1_with_keeper/mappo.yaml \
        --checkpoint_dir logs/.../agent_0/default-1/epoch_500

    # Evaluate best and last only, with more games
    python scripts/eval_checkpoint.py \
        --config expr_configs/cooperative_MARL_benchmark/academy/3_vs_1_with_keeper/mappo.yaml \
        --run_dir logs/gr_football/benchmark_academy_3_vs_1_with_keeper_mappo/2026-03-14-12-00-00 \
        --filter best last --num_games 200

    # Only evaluate every Nth epoch checkpoint
    python scripts/eval_checkpoint.py \
        --config ... --run_dir ... --epoch_interval 100
"""

import argparse
import os
import sys
import re
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from multiprocessing import get_context

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, BASE_DIR)

from light_malib.algorithm.mappo.policy import MAPPO
from light_malib.envs.gr_football.env import GRFootballEnv
from light_malib.rollout.rollout_func import rollout_func
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.utils.cfg import load_cfg
from light_malib.utils.logger import Logger


def discover_checkpoints(run_dir, agent_id="agent_0", epoch_interval=None, filter_names=None):
    """
    Find all checkpoint directories under a run directory.

    Expected layout:
        run_dir/agent_0/<policy_id>/<checkpoint_name>/
    where checkpoint_name is like epoch_100, best, 500.last, etc.
    """
    agent_dir = os.path.join(run_dir, agent_id)
    if not os.path.isdir(agent_dir):
        print(f"[ERROR] Agent directory not found: {agent_dir}")
        sys.exit(1)

    checkpoints = []
    for policy_id in os.listdir(agent_dir):
        policy_dir = os.path.join(agent_dir, policy_id)
        if not os.path.isdir(policy_dir):
            continue
        for ckpt_name in os.listdir(policy_dir):
            ckpt_dir = os.path.join(policy_dir, ckpt_name)
            if not os.path.isdir(ckpt_dir):
                continue
            desc_file = os.path.join(ckpt_dir, "desc.pkl")
            if not os.path.exists(desc_file):
                continue
            checkpoints.append((ckpt_name, ckpt_dir))

    if filter_names:
        checkpoints = [
            (name, path) for name, path in checkpoints
            if any(f in name for f in filter_names)
        ]

    def sort_key(item):
        name = item[0]
        if name == "best":
            return (2, 0)
        m = re.search(r"(\d+)", name)
        epoch = int(m.group(1)) if m else 0
        if "last" in name:
            return (1, epoch)
        return (0, epoch)

    checkpoints.sort(key=sort_key)

    if epoch_interval and epoch_interval > 1:
        filtered = []
        for name, path in checkpoints:
            m = re.match(r"epoch_(\d+)$", name)
            if m:
                epoch = int(m.group(1))
                if epoch % epoch_interval != 0:
                    continue
            filtered.append((name, path))
        checkpoints = filtered

    return checkpoints


def _worker_eval(args):
    """Worker function: each process creates its own env + policies and runs games."""
    worker_id, ckpt_dir, opponent_dir, env_cfg, rollout_length, games_for_worker = args

    policy_0 = MAPPO.load(ckpt_dir, env_agent_id="agent_0")
    policy_1 = MAPPO.load(opponent_dir, env_agent_id="agent_1")
    policy_0.eval()
    policy_1.eval()

    env = GRFootballEnv(worker_id, None, env_cfg)
    rollout_desc = RolloutDesc("agent_0", None, None, None, None, None)
    behavior_policies = {
        "agent_0": ("policy_0", policy_0),
        "agent_1": ("policy_1", policy_1),
    }

    stats = []
    for _ in range(games_for_worker):
        rollout_results = rollout_func(
            eval=True,
            rollout_worker=None,
            rollout_desc=rollout_desc,
            env=env,
            behavior_policies=behavior_policies,
            data_server=None,
            rollout_length=rollout_length,
            render=False,
            rollout_epoch=100,
        )
        for result in rollout_results["results"]:
            stats.append(result["stats"]["agent_0"])

    env._env.close()
    return stats


def evaluate_checkpoint(ckpt_dir, cfg, opponent_dir, num_games, rollout_length, num_workers=1):
    """Run num_games episodes across num_workers parallel processes."""
    env_cfg = cfg.rollout_manager.worker.envs[0]
    # Pass env_cfg as-is so GRFootballEnv gets attribute access (e.g. .reward_config)

    if num_workers <= 1:
        return _worker_eval((0, ckpt_dir, opponent_dir, env_cfg, rollout_length, num_games))

    # Distribute games evenly across workers
    base, extra = divmod(num_games, num_workers)
    worker_args = []
    for i in range(num_workers):
        n = base + (1 if i < extra else 0)
        if n > 0:
            worker_args.append((i, ckpt_dir, opponent_dir, env_cfg, rollout_length, n))

    all_stats = []
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(worker_args), mp_context=ctx) as pool:
        futures = [pool.submit(_worker_eval, wa) for wa in worker_args]
        for future in as_completed(futures):
            all_stats.extend(future.result())

    return all_stats


def compute_ci(values, confidence=0.95):
    """Compute mean and confidence interval using the normal approximation."""
    n = len(values)
    mean = np.mean(values)
    if n < 2:
        return mean, 0.0
    se = np.std(values, ddof=1) / np.sqrt(n)
    from scipy import stats as sp_stats
    z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)
    return mean, z * se


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints against built-in AI")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Path to a run directory containing checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Path to a single checkpoint to evaluate")
    parser.add_argument("--opponent", type=str,
                        default="light_malib/trained_models/gr_football/5_vs_5/built_in",
                        help="Path to opponent model")
    parser.add_argument("--num_games", type=int, default=100,
                        help="Number of games per checkpoint (default: 100)")
    parser.add_argument("--agent_id", type=str, default="agent_0")
    parser.add_argument("--filter", nargs="*", default=None,
                        help="Filter checkpoint names (e.g. best last epoch_500)")
    parser.add_argument("--epoch_interval", type=int, default=None,
                        help="Only evaluate every Nth epoch checkpoint")
    _ncpu = (os.cpu_count() or 8) - 2
    _default_workers = min(32, max(1, _ncpu))
    parser.add_argument("--num_workers", type=int, default=_default_workers,
                        help="Number of parallel workers per checkpoint (default: min(32, cpu_count-2))")
    parser.add_argument("--parallel_checkpoints", type=int, default=1,
                        help="Evaluate N checkpoints at a time (default: 1). Increases total CPU use.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (default: results/eval_<run_name>)")
    args = parser.parse_args()

    if args.run_dir is None and args.checkpoint_dir is None:
        print("[ERROR] Must specify either --run_dir or --checkpoint_dir")
        sys.exit(1)

    config_path = os.path.join(BASE_DIR, args.config)
    cfg = load_cfg(config_path)
    cfg["rollout_manager"]["worker"]["envs"][0]["scenario_config"]["render"] = False
    rollout_length = cfg.rollout_manager.worker.eval_rollout_length

    opponent_dir = os.path.join(BASE_DIR, args.opponent)

    if args.checkpoint_dir:
        checkpoints = [("custom", args.checkpoint_dir)]
    else:
        checkpoints = discover_checkpoints(
            args.run_dir, args.agent_id, args.epoch_interval, args.filter
        )

    if not checkpoints:
        print("[ERROR] No checkpoints found")
        sys.exit(1)

    print(f"Found {len(checkpoints)} checkpoint(s) to evaluate:")
    for name, path in checkpoints:
        print(f"  - {name}: {path}")

    if args.output_dir is None:
        if args.run_dir:
            args.output_dir = args.run_dir
        else:
            args.output_dir = os.path.dirname(args.checkpoint_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    parallel_ckpts = max(1, min(args.parallel_checkpoints, len(checkpoints)))
    workers_per_ckpt = max(1, args.num_workers // parallel_ckpts)
    if parallel_ckpts > 1:
        print(f"Using {parallel_ckpts} checkpoint(s) in parallel, {workers_per_ckpt} workers each")

    def run_one_checkpoint(item):
        idx, ckpt_name, ckpt_dir = item
        w = min(workers_per_ckpt, args.num_games)
        t0 = time.time()
        stats_list = evaluate_checkpoint(ckpt_dir, cfg, opponent_dir, args.num_games, rollout_length, w)
        elapsed = time.time() - t0
        wins = [s["win"] for s in stats_list]
        rewards = [s["reward"] for s in stats_list]
        goals = [s["my_goal"] for s in stats_list]
        scores = [s["score"] for s in stats_list]
        win_mean, win_ci = compute_ci(wins)
        reward_mean, reward_ci = compute_ci(rewards)
        goal_mean, goal_ci = compute_ci(goals)
        m = re.search(r"(\d+)", ckpt_name)
        epoch = int(m.group(1)) if m else -1
        return (
            idx,
            ckpt_name,
            elapsed,
            len(stats_list),
            {
                "checkpoint": ckpt_name,
                "epoch": epoch,
                "win_rate": win_mean,
                "win_ci": win_ci,
                "reward_mean": reward_mean,
                "reward_ci": reward_ci,
                "goal_mean": goal_mean,
                "goal_ci": goal_ci,
                "score_mean": np.mean(scores),
                "num_games": len(wins),
            },
            (win_mean, reward_mean, goal_mean),
        )

    results = [None] * len(checkpoints)
    if parallel_ckpts <= 1:
        for idx, (ckpt_name, ckpt_dir) in enumerate(checkpoints):
            print(f"\n[{idx+1}/{len(checkpoints)}] Evaluating {ckpt_name} ({args.num_games} games, {workers_per_ckpt} workers)...")
            _, _, elapsed, n, result, _ = run_one_checkpoint((idx, ckpt_name, ckpt_dir))
            results[idx] = result
            print(f"  Completed in {elapsed:.1f}s ({elapsed/n:.2f}s/game)")
            print(f"  Win Rate:  {result['win_rate']:.3f} ± {result['win_ci']:.3f}")
            print(f"  Reward:    {result['reward_mean']:.3f} ± {result['reward_ci']:.3f}")
            print(f"  Goals:     {result['goal_mean']:.3f} ± {result['goal_ci']:.3f}")
    else:
        items = [(idx, name, path) for idx, (name, path) in enumerate(checkpoints)]
        with ThreadPoolExecutor(max_workers=parallel_ckpts) as executor:
            futures = [executor.submit(run_one_checkpoint, item) for item in items]
            for future in as_completed(futures):
                idx, ckpt_name, elapsed, n, result, (wr, rm, gm) = future.result()
                results[idx] = result
                print(f"\n[{idx+1}/{len(checkpoints)}] {ckpt_name}: {elapsed:.1f}s ({elapsed/n:.2f}s/game) | Win {wr:.3f} Reward {rm:.3f}")
        results = [r for r in results if r is not None]

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "eval_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Checkpoint':>20s} | {'Win Rate':>12s} | {'Reward':>14s} | {'Goals':>12s} | {'Games':>5s}")
    print("-" * 80)
    for r in results:
        print(f"{r['checkpoint']:>20s} | {r['win_rate']:.3f} ± {r['win_ci']:.3f} | "
              f"{r['reward_mean']:.3f} ± {r['reward_ci']:.3f} | "
              f"{r['goal_mean']:.3f} ± {r['goal_ci']:.3f} | {r['num_games']:>5d}")
    print("=" * 80)

    # Plot win rate over checkpoints if multiple
    epoch_results = [r for r in results if r["epoch"] >= 0]
    if len(epoch_results) > 1:
        epoch_results.sort(key=lambda r: r["epoch"])
        epochs = [r["epoch"] for r in epoch_results]
        win_rates = [r["win_rate"] for r in epoch_results]
        win_cis = [r["win_ci"] for r in epoch_results]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(epochs, win_rates, yerr=win_cis, fmt="o-", capsize=4,
                     linewidth=1.5, markersize=5, label="Win Rate ± 95% CI")
        ax.set_xlabel("Training Epoch")
        ax.set_ylabel("Win Rate")
        ax.set_title("Win Rate vs Training Epoch (Checkpoint Evaluation)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig_path = os.path.join(args.output_dir, "eval_win_rate.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nPlot saved to {fig_path}")


if __name__ == "__main__":
    main()
