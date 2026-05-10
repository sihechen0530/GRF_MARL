#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

BASE_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, BASE_DIR)

from light_malib.algorithm.mappo.policy import MAPPO
from light_malib.llm.entropy_mask import EntropyGuidedMask
from light_malib.rollout.rollout_func import env_reset, rename_fields, select_fields, update_fields
from light_malib.utils.cfg import load_cfg
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.utils.episode import EpisodeKey


DEFAULT_BASELINE_RUN = "logs/gr_football/benchmark_academy_counterattack_mappo/2026-04-12-15-50-48"
DEFAULT_OURS_RUN = "logs/gr_football/llm_mask_academy_counterattack_mappo/2026-03-23-18-32-07"
DEFAULT_CONFIG = "expr_configs/cooperative_MARL_benchmark/academy/counterattack/mappo.yaml"
DEFAULT_OPPONENT = "light_malib/trained_models/gr_football/11_vs_11/built_in"


def read_eval_rows(csv_path: str):
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(float(row["epoch"]))
                win_rate = float(row["win_rate"])
            except (KeyError, ValueError):
                continue
            if epoch < 0:
                continue
            row["_epoch"] = epoch
            row["_win_rate"] = win_rate
            rows.append(row)
    return rows


def resolve_checkpoint(run_dir: str):
    csv_path = os.path.join(run_dir, "eval_results.csv")
    rows = read_eval_rows(csv_path)
    agent_dir = os.path.join(run_dir, "agent_0")
    if not os.path.isdir(agent_dir):
        raise FileNotFoundError(f"agent dir not found under {run_dir}")
    policy_ids = [x for x in os.listdir(agent_dir) if os.path.isdir(os.path.join(agent_dir, x))]
    if not policy_ids:
        raise FileNotFoundError(f"no policy ids found under {agent_dir}")
    policy_id = policy_ids[0]
    policy_dir = os.path.join(agent_dir, policy_id)

    if rows:
        best = max(rows, key=lambda row: (row["_win_rate"], row["_epoch"]))
        ckpt_name = best["checkpoint"]
        ckpt_dir = os.path.join(policy_dir, ckpt_name)
        if os.path.isdir(ckpt_dir):
            return ckpt_dir, {"checkpoint": ckpt_name, "epoch": best["_epoch"], "win_rate": best["_win_rate"]}

        epoch_dir = os.path.join(policy_dir, f"epoch_{best['_epoch']}")
        if os.path.isdir(epoch_dir):
            return epoch_dir, {"checkpoint": f"epoch_{best['_epoch']}", "epoch": best["_epoch"], "win_rate": best["_win_rate"]}

    for fallback in ("best", "last"):
        ckpt_dir = os.path.join(policy_dir, fallback)
        if os.path.isdir(ckpt_dir):
            return ckpt_dir, {"checkpoint": fallback, "epoch": None, "win_rate": None}

    candidates = [x for x in os.listdir(policy_dir) if os.path.isdir(os.path.join(policy_dir, x))]
    if not candidates:
        raise FileNotFoundError(f"no checkpoints found under {policy_dir}")
    candidates.sort()
    ckpt_name = candidates[-1]
    return os.path.join(policy_dir, ckpt_name), {"checkpoint": ckpt_name, "epoch": None, "win_rate": None}


def js_divergence(p: np.ndarray, q: np.ndarray):
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * (np.log(p) - np.log(m)), axis=-1) + 0.5 * np.sum(q * (np.log(q) - np.log(m)), axis=-1)


def compute_masked_probs(policy: MAPPO, policy_input: dict):
    observations = policy_input[EpisodeKey.CUR_OBS]
    actor_rnn_states = policy_input[EpisodeKey.ACTOR_RNN_STATE]
    critic_rnn_states = policy_input[EpisodeKey.CRITIC_RNN_STATE]
    action_masks = policy_input[EpisodeKey.ACTION_MASK]
    rnn_masks = policy_input[EpisodeKey.DONE]
    states = policy_input.get(EpisodeKey.CUR_STATE, observations)

    with torch.no_grad():
        if policy.share_backbone:
            observations = policy.backbone(states, observations, critic_rnn_states, rnn_masks)

        raw_logits, _ = policy.actor.logits(observations, actor_rnn_states, rnn_masks)
        action_masks_t = torch.as_tensor(action_masks, dtype=torch.float32)
        masked_logits = raw_logits - 1e10 * (1 - action_masks_t)
        probs = torch.distributions.Categorical(logits=masked_logits).probs.detach().cpu().numpy()
    return probs


def policy_inputs_from_step(step_data: dict):
    return rename_fields(
        copy.deepcopy(step_data),
        [EpisodeKey.NEXT_OBS, EpisodeKey.NEXT_STATE],
        [EpisodeKey.CUR_OBS, EpisodeKey.CUR_OBS],
    )


def regime_key(obs: np.ndarray, action_mask: np.ndarray):
    regime = EntropyGuidedMask.extract_regime(obs, action_mask)
    return (
        int(regime["game_mode"]),
        int(regime["possession"]),
        int(regime["ball_zone"]),
        int(regime.get("player_zone", 0)),
    )


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def run_driver_rollouts(env, driver_name, driver_policy, baseline_policy, ours_policy, opponent_policy, rollout_length, num_episodes):
    behavior_policies = {
        "agent_0": ("driver", driver_policy),
        "agent_1": ("opponent", opponent_policy),
    }
    custom_reset_config = {
        "feature_encoders": {
            "agent_0": driver_policy.feature_encoder,
            "agent_1": opponent_policy.feature_encoder,
        },
        "main_agent_id": "agent_0",
        "rollout_length": rollout_length,
        "rollout_epoch": 0,
        "eval": True,
    }
    rollout_desc = RolloutDesc("agent_0", None, None, None, None, None)

    per_step_js = []
    per_step_agreement = []
    per_step_entropy_gap = []
    regime_stats = defaultdict(lambda: {"js": [], "agreement": []})

    for _ in range(num_episodes):
        step_data = env_reset(env, behavior_policies, custom_reset_config)
        step = 0

        while step <= rollout_length and not env.is_terminated():
            policy_inputs = policy_inputs_from_step(step_data)
            driver_input = policy_inputs["agent_0"]

            base_probs = compute_masked_probs(baseline_policy, driver_input)
            ours_probs = compute_masked_probs(ours_policy, driver_input)

            js_vals = js_divergence(base_probs, ours_probs)
            base_argmax = np.argmax(base_probs, axis=-1)
            ours_argmax = np.argmax(ours_probs, axis=-1)
            agreement = (base_argmax == ours_argmax).astype(np.float32)

            base_entropy = -(base_probs * np.log(np.clip(base_probs, 1e-12, 1.0))).sum(axis=-1)
            ours_entropy = -(ours_probs * np.log(np.clip(ours_probs, 1e-12, 1.0))).sum(axis=-1)
            entropy_gap = np.abs(base_entropy - ours_entropy)

            per_step_js.append(float(js_vals.mean()))
            per_step_agreement.append(float(agreement.mean()))
            per_step_entropy_gap.append(float(entropy_gap.mean()))

            obs_batch = driver_input[EpisodeKey.CUR_OBS]
            mask_batch = driver_input[EpisodeKey.ACTION_MASK]
            for idx in range(obs_batch.shape[0]):
                key = regime_key(obs_batch[idx], mask_batch[idx])
                regime_stats[key]["js"].append(float(js_vals[idx]))
                regime_stats[key]["agreement"].append(float(agreement[idx]))

            policy_outputs = {}
            for agent_id, (_, policy) in behavior_policies.items():
                policy_outputs[agent_id] = policy.compute_action(
                    inference=True,
                    explore=False,
                    to_numpy=True,
                    **policy_inputs[agent_id],
                )

            actions = select_fields(policy_outputs, [EpisodeKey.ACTION])
            env_rets = env.step(actions)
            step_data = update_fields(
                env_rets,
                select_fields(policy_outputs, [EpisodeKey.ACTOR_RNN_STATE, EpisodeKey.CRITIC_RNN_STATE]),
            )
            step += 1

    top_regimes = []
    for key, values in regime_stats.items():
        top_regimes.append(
            {
                "regime": list(key),
                "count": len(values["js"]),
                "mean_js": float(np.mean(values["js"])),
                "mean_agreement": float(np.mean(values["agreement"])),
            }
        )
    top_regimes.sort(key=lambda row: (-row["mean_js"], -row["count"]))

    return {
        "driver": driver_name,
        "js_summary": summarize(per_step_js),
        "agreement_summary": summarize(per_step_agreement),
        "entropy_gap_summary": summarize(per_step_entropy_gap),
        "js_values": per_step_js,
        "agreement_values": per_step_agreement,
        "entropy_gap_values": per_step_entropy_gap,
        "top_regimes": top_regimes[:12],
    }


def plot_histogram(results_by_driver, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [
        ("js_values", "JS Divergence"),
        ("agreement_values", "Argmax Agreement"),
        ("entropy_gap_values", "Entropy Gap"),
    ]

    for ax, (metric_key, title) in zip(axes, metrics):
        for name, color in (("baseline", "#1f77b4"), ("ours", "#d95f02")):
            vals = np.asarray(results_by_driver[name][metric_key], dtype=np.float32)
            if vals.size == 0:
                continue
            ax.hist(vals, bins=30, alpha=0.55, color=color, label=f"{name}-driven")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle("Counterattack MAPPO: Baseline vs LLM-Mask Policy Divergence", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_regime_bars(results_by_driver, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for ax, driver_name in zip(axes, ("baseline", "ours")):
        rows = results_by_driver[driver_name]["top_regimes"][:8]
        labels = [f"{tuple(row['regime'])}\nN={row['count']}" for row in rows]
        vals = [row["mean_js"] for row in rows]
        ax.bar(np.arange(len(rows)), vals, color="#4c78a8")
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Mean JS")
        ax.set_title(f"Top Divergent Regimes on {driver_name}-driven states")
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Policy Divergence by Regime\n(game_mode, possession, ball_zone, player_zone)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare learned policies on matched states")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--baseline_run_dir", default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--ours_run_dir", default=DEFAULT_OURS_RUN)
    parser.add_argument("--opponent", default=DEFAULT_OPPONENT)
    parser.add_argument("--num_episodes", type=int, default=6)
    parser.add_argument("--rollout_length", type=int, default=None)
    args = parser.parse_args()

    try:
        from light_malib.envs.gr_football.env import GRFootballEnv
    except Exception as exc:
        print(
            "Could not import GRFootballEnv. This script requires the Google Football "
            f"environment (`gfootball`) to be installed and importable. Original error: {exc}"
        )
        return

    cfg = load_cfg(os.path.join(BASE_DIR, args.config))
    cfg["rollout_manager"]["worker"]["envs"][0]["scenario_config"]["render"] = False
    rollout_length = args.rollout_length or cfg.rollout_manager.worker.eval_rollout_length

    baseline_ckpt, baseline_meta = resolve_checkpoint(os.path.join(BASE_DIR, args.baseline_run_dir))
    ours_ckpt, ours_meta = resolve_checkpoint(os.path.join(BASE_DIR, args.ours_run_dir))

    baseline_policy = MAPPO.load(baseline_ckpt, env_agent_id="agent_0")
    ours_policy = MAPPO.load(ours_ckpt, env_agent_id="agent_0")
    opponent_policy = MAPPO.load(os.path.join(BASE_DIR, args.opponent), env_agent_id="agent_1")
    baseline_policy.eval()
    ours_policy.eval()
    opponent_policy.eval()

    env_cfg = cfg.rollout_manager.worker.envs[0]
    env = GRFootballEnv(0, None, env_cfg)

    results_by_driver = {
        "baseline": run_driver_rollouts(env, "baseline", baseline_policy, baseline_policy, ours_policy, opponent_policy, rollout_length, args.num_episodes),
        "ours": run_driver_rollouts(env, "ours", ours_policy, baseline_policy, ours_policy, opponent_policy, rollout_length, args.num_episodes),
    }
    env._env.close()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_DIR, "results", f"policy_divergence_counterattack_mappo_{timestamp}")
    os.makedirs(output_dir, exist_ok=False)

    hist_path = os.path.join(output_dir, "distribution_divergence_histograms.png")
    regime_path = os.path.join(output_dir, "divergence_by_regime.png")
    plot_histogram(results_by_driver, hist_path)
    plot_regime_bars(results_by_driver, regime_path)

    summary = {
        "output_dir": output_dir,
        "baseline_checkpoint": {"path": baseline_ckpt, **baseline_meta},
        "ours_checkpoint": {"path": ours_ckpt, **ours_meta},
        "opponent": os.path.join(BASE_DIR, args.opponent),
        "num_episodes_per_driver": args.num_episodes,
        "rollout_length": rollout_length,
        "results_by_driver": results_by_driver,
        "plots": {
            "histograms": hist_path,
            "regime_bars": regime_path,
        },
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved policy divergence report to {output_dir}")


if __name__ == "__main__":
    main()
