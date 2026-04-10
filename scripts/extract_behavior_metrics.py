#!/usr/bin/env python3
"""
Extract behavioral metrics from eval episodes for LLM-based reward diagnostics.

Runs evaluation games with a trained checkpoint and records per-step raw GRF
observations.  Aggregates them into a structured metrics dict suitable for
feeding to an LLM diagnostic prompt (iterative Eureka).

Usage:
  python scripts/extract_behavior_metrics.py \
      --checkpoint logs/gr_football/<expr>/<timestamp> \
      --config expr_configs/.../mappo_eureka_llm.yaml \
      --num-games 20 \
      --output metrics/behavior_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from light_malib.envs.gr_football.reward_state import extract_reward_state


# ---------------------------------------------------------------------------
# Checkpoint path resolution
# ---------------------------------------------------------------------------
def resolve_policy_checkpoint(path: str, agent_id: str = "agent_0") -> str:
    """Resolve a run directory or checkpoint path to an actual policy dir with desc.pkl.

    Accepted inputs:
      - Direct policy dir (has desc.pkl)                  → returned as-is
      - Run dir (timestamp):  <run>/agent_0/<pid>/best/   → best or latest epoch
      - Experiment dir:       <exp>/<latest_run>/agent_0/… → walks down
    """
    import os, re

    # Already a policy checkpoint?
    if os.path.isfile(os.path.join(path, "desc.pkl")):
        return path

    # Try run_dir/agent_0/<policy_id>/<ckpt>/
    agent_dir = os.path.join(path, agent_id)
    if os.path.isdir(agent_dir):
        for policy_id in sorted(os.listdir(agent_dir)):
            policy_dir = os.path.join(agent_dir, policy_id)
            if not os.path.isdir(policy_dir):
                continue
            # prefer "best", then highest epoch_N, then anything with desc.pkl
            candidates = []
            for name in os.listdir(policy_dir):
                ckpt = os.path.join(policy_dir, name)
                if os.path.isfile(os.path.join(ckpt, "desc.pkl")):
                    candidates.append((name, ckpt))
            if not candidates:
                continue

            # sort: best first, then by epoch number descending
            def _sort_key(item):
                n = item[0]
                if n == "best":
                    return (0, 999999)
                m = re.search(r"(\d+)", n)
                epoch = int(m.group(1)) if m else -1
                if "last" in n:
                    return (1, epoch)
                return (2, -epoch)

            candidates.sort(key=_sort_key)
            chosen = candidates[0][1]
            print(f"  Resolved checkpoint: {chosen}")
            return chosen

    # Maybe it's an experiment dir — pick the latest timestamped run
    if os.path.isdir(path):
        subdirs = sorted([
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ])
        if subdirs:
            latest_run = os.path.join(path, subdirs[-1])
            return resolve_policy_checkpoint(latest_run, agent_id)

    raise FileNotFoundError(
        f"Could not find desc.pkl under {path}. "
        f"Expected structure: <run_dir>/{agent_id}/<policy_id>/<checkpoint>/"
    )


# ---------------------------------------------------------------------------
# Field geometry constants (GRF coordinate system)
# ---------------------------------------------------------------------------
FIELD_X_MIN, FIELD_X_MAX = -1.0, 1.0
FIELD_Y_MIN, FIELD_Y_MAX = -0.42, 0.42
THIRD_X = (FIELD_X_MAX - FIELD_X_MIN) / 3.0  # ~0.667
DEF_THIRD_X = FIELD_X_MIN + THIRD_X           # ~ -0.333
ATT_THIRD_X = FIELD_X_MAX - THIRD_X           # ~ +0.333
PENALTY_X = 0.64
PENALTY_Y = 0.27


def _nan_safe(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return 0.0
    return v


# ---------------------------------------------------------------------------
# Per-step snapshot collector (runs inside each eval worker)
# ---------------------------------------------------------------------------
class StepRecorder:
    """Lightweight per-step recorder that stores raw GRF obs fields we need."""

    def __init__(self, n_left_controlled: int):
        self.n_left = n_left_controlled
        self.steps: list[dict] = []

    def record(self, obs_list: list, actions: np.ndarray | None = None):
        """obs_list: list of raw GRF obs dicts, one per on-pitch player."""
        obs0 = obs_list[0]
        snap = {
            "ball": np.array(obs0["ball"], dtype=np.float32).tolist(),
            "ball_owned_team": int(obs0["ball_owned_team"]),
            "ball_owned_player": int(obs0["ball_owned_player"]),
            "left_team": np.array(obs0["left_team"], dtype=np.float32).tolist(),
            "right_team": np.array(obs0["right_team"], dtype=np.float32).tolist(),
            "game_mode": int(obs0["game_mode"]),
            "score": [int(s) for s in obs0["score"]],
            "steps_left": int(obs0["steps_left"]),
            "active": int(obs0["active"]),
        }
        if actions is not None:
            snap["actions"] = actions[: self.n_left].tolist()
        self.steps.append(snap)


# ---------------------------------------------------------------------------
# Metric computation from recorded steps
# ---------------------------------------------------------------------------
def compute_episode_metrics(steps: list[dict], n_left_ctrl: int) -> dict:
    """Compute comprehensive behavioral metrics from one episode's step records."""
    T = len(steps)
    if T == 0:
        return {}

    n_left_total = len(steps[0]["left_team"])
    n_right_total = len(steps[0]["right_team"])

    # --- basic accumulators ---
    possession_ours = 0
    possession_theirs = 0
    ball_x_sum = 0.0
    territory = {"def": 0, "mid": 0, "att": 0}

    # per-agent trackers (for controlled left agents, indices 0..n_left_ctrl-1)
    agent_positions = [[] for _ in range(n_left_ctrl)]
    agent_touches = [0] * n_left_ctrl
    action_counts = defaultdict(int)

    # formation trackers
    team_widths = []
    team_depths = []
    team_compactness = []

    # defensive trackers
    def_line_heights = []
    press_dists = []

    # ball progression
    ball_x_series = []

    for t, snap in enumerate(steps):
        ball_x, ball_y = snap["ball"][0], snap["ball"][1]
        ball_x_series.append(ball_x)
        ball_x_sum += ball_x
        bot = snap["ball_owned_team"]
        bop = snap["ball_owned_player"]

        # --- possession ---
        if bot == 0:
            possession_ours += 1
        elif bot == 1:
            possession_theirs += 1

        # --- territory ---
        if ball_x < DEF_THIRD_X:
            territory["def"] += 1
        elif ball_x > ATT_THIRD_X:
            territory["att"] += 1
        else:
            territory["mid"] += 1

        # --- per-agent position & touches ---
        left_pos = np.array(snap["left_team"], dtype=np.float32)
        for i in range(min(n_left_ctrl, len(left_pos))):
            agent_positions[i].append(left_pos[i].tolist())
            if bot == 0 and bop == i:
                agent_touches[i] += 1

        # --- action distribution ---
        if "actions" in snap:
            for a in snap["actions"]:
                action_counts[int(a)] += 1

        # --- formation: controlled players only (skip GK at index 0 if n_left_total > n_left_ctrl) ---
        ctrl_pos = left_pos[:n_left_ctrl]
        if len(ctrl_pos) >= 2:
            xs = ctrl_pos[:, 0]
            ys = ctrl_pos[:, 1]
            team_widths.append(float(ys.max() - ys.min()))
            team_depths.append(float(xs.max() - xs.min()))
            dists = np.linalg.norm(
                ctrl_pos[:, None, :] - ctrl_pos[None, :, :], axis=-1
            )
            avg_dist = dists.sum() / max(len(ctrl_pos) * (len(ctrl_pos) - 1), 1)
            team_compactness.append(float(avg_dist))

        # --- defensive line height (lowest x among non-GK controlled) ---
        if n_left_ctrl >= 2:
            def_line_heights.append(float(ctrl_pos[1:, 0].min()) if len(ctrl_pos) > 1 else 0.0)

        # --- pressing intensity (avg dist to ball when opponent has ball) ---
        if bot == 1:
            ball_xy = np.array([ball_x, ball_y])
            dists_to_ball = np.linalg.norm(ctrl_pos[:, :2] - ball_xy, axis=1)
            press_dists.append(float(dists_to_ball.mean()))

    # --- aggregate ---
    total_possession = possession_ours + possession_theirs
    final_score = steps[-1]["score"]

    metrics: dict = {
        "episode_length": T,
        "final_score_ours": final_score[0],
        "final_score_theirs": final_score[1],
        "result": "win" if final_score[0] > final_score[1]
                  else ("loss" if final_score[0] < final_score[1] else "draw"),

        # Team-level
        "possession_rate": possession_ours / max(total_possession, 1),
        "territory_pct": {
            k: v / max(T, 1) for k, v in territory.items()
        },
        "avg_ball_x": ball_x_sum / T,
        "ball_progression": float(np.mean(np.diff(ball_x_series))) if T > 1 else 0.0,

        # Formation
        "avg_team_width": float(np.mean(team_widths)) if team_widths else 0.0,
        "avg_team_depth": float(np.mean(team_depths)) if team_depths else 0.0,
        "avg_compactness": float(np.mean(team_compactness)) if team_compactness else 0.0,

        # Defense
        "avg_def_line_height": float(np.mean(def_line_heights)) if def_line_heights else 0.0,
        "avg_press_distance": float(np.mean(press_dists)) if press_dists else 0.0,

        # Per-agent
        "agent_touches": agent_touches,
        "agent_avg_positions": [],

        # Action distribution
        "action_distribution": {},
    }

    for i in range(n_left_ctrl):
        if agent_positions[i]:
            arr = np.array(agent_positions[i])
            metrics["agent_avg_positions"].append({
                "agent": i,
                "avg_x": float(arr[:, 0].mean()),
                "avg_y": float(arr[:, 1].mean()),
                "std_x": float(arr[:, 0].std()),
                "std_y": float(arr[:, 1].std()),
            })

    total_actions = sum(action_counts.values()) or 1
    ACTION_NAMES = {
        0: "idle", 1: "left", 2: "top_left", 3: "top", 4: "top_right",
        5: "right", 6: "bottom_right", 7: "bottom", 8: "bottom_left",
        9: "long_pass", 10: "high_pass", 11: "short_pass", 12: "shot",
        13: "sprint", 14: "release_direction", 15: "release_sprint",
        16: "slide", 17: "dribble", 18: "release_dribble",
    }
    for a_id, cnt in sorted(action_counts.items()):
        name = ACTION_NAMES.get(a_id, f"action_{a_id}")
        metrics["action_distribution"][name] = round(cnt / total_actions, 4)

    return metrics


# ---------------------------------------------------------------------------
# Worker: run eval games and collect per-step data
# ---------------------------------------------------------------------------
def _worker_eval(args) -> list[dict]:
    """Run games in a subprocess and return per-episode metrics."""
    (
        ckpt_dir,
        opponent_dir,
        env_cfg,
        rollout_length,
        num_games,
        worker_id,
        n_left_ctrl,
        phi_module_path,
    ) = args

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    from omegaconf import OmegaConf
    from light_malib.algorithm.mappo.policy import MAPPO
    from light_malib.envs.gr_football.env import GRFootballEnv
    from light_malib.utils.desc.task_desc import RolloutDesc
    from light_malib.utils.episode import EpisodeKey
    from collections import OrderedDict

    # env_cfg arrives as a plain dict (for pickling); GRFootballEnv needs attribute access
    if isinstance(env_cfg, dict):
        env_cfg = OmegaConf.create(env_cfg)

    policy_0 = MAPPO.load(ckpt_dir, env_agent_id="agent_0")
    policy_0.eval()
    if opponent_dir:
        policy_1 = MAPPO.load(opponent_dir, env_agent_id="agent_1")
    else:
        policy_1 = MAPPO.load(
            "light_malib/trained_models/gr_football/5_vs_5/built_in",
            env_agent_id="agent_1",
        )
    policy_1.eval()

    env = GRFootballEnv(worker_id, None, env_cfg)
    behavior_policies = OrderedDict([
        ("agent_0", ("policy_0", policy_0)),
        ("agent_1", ("policy_1", policy_1)),
    ])

    # optionally load phi for phi-diagnostic metrics
    phi_fn = None
    if phi_module_path:
        import importlib
        try:
            mod = importlib.import_module(phi_module_path)
            phi_fn = getattr(mod, "phi", None)
        except Exception:
            pass

    all_episode_metrics = []

    for game_i in range(num_games):
        feature_encoders = OrderedDict()
        for aid, (pid, pol) in behavior_policies.items():
            feature_encoders[aid] = pol.feature_encoder

        custom_reset_config = {
            "feature_encoders": feature_encoders,
            "main_agent_id": "agent_0",
            "rollout_length": rollout_length,
        }
        env_rets = env.reset(custom_reset_config)

        recorder = StepRecorder(n_left_ctrl)
        # record initial obs
        recorder.record([s.obs for s in env.states])

        step_data = env_rets
        phi_values = []

        # initialize RNN hidden states for each policy
        rnn_states = {}
        for aid, (pid, pol) in behavior_policies.items():
            obs_sample = step_data[aid].get(EpisodeKey.CUR_OBS,
                         step_data[aid].get(EpisodeKey.NEXT_OBS))
            batch_size = obs_sample.shape[0] if hasattr(obs_sample, "shape") else 1
            rnn_states[aid] = pol.get_initial_state(batch_size)

        for step_i in range(rollout_length):
            # build policy inputs
            policy_inputs = {}
            for aid in behavior_policies:
                d = dict(step_data[aid])
                if EpisodeKey.NEXT_OBS in d:
                    d[EpisodeKey.CUR_OBS] = d.pop(EpisodeKey.NEXT_OBS)
                # inject RNN states
                d[EpisodeKey.ACTOR_RNN_STATE] = rnn_states[aid][EpisodeKey.ACTOR_RNN_STATE]
                d[EpisodeKey.CRITIC_RNN_STATE] = rnn_states[aid][EpisodeKey.CRITIC_RNN_STATE]
                policy_inputs[aid] = d

            # compute actions
            policy_outputs = {}
            for aid, (pid, pol) in behavior_policies.items():
                policy_outputs[aid] = pol.compute_action(
                    inference=True, explore=False, to_numpy=True,
                    **policy_inputs[aid],
                )

            actions_dict = {
                aid: {EpisodeKey.ACTION: policy_outputs[aid][EpisodeKey.ACTION]}
                for aid in behavior_policies
            }
            env_rets = env.step(actions_dict)

            all_actions = np.concatenate([
                policy_outputs[aid][EpisodeKey.ACTION]
                for aid in behavior_policies
            ], axis=0).flatten()
            recorder.record([s.obs for s in env.states], all_actions)

            # phi diagnostics
            if phi_fn is not None:
                obs0 = env.states[0].obs
                n_lp = len(obs0["left_team_roles"])
                n_rp = len(obs0["right_team_roles"])
                for pi in range(n_left_ctrl):
                    try:
                        rs = extract_reward_state(obs0, pi, n_lp, n_rp)
                        pv = phi_fn(rs, "default")
                        phi_values.append(float(pv))
                    except Exception:
                        pass

            # update RNN states and env data for next step
            step_data = {}
            for aid in behavior_policies:
                d = dict(env_rets[aid])
                if EpisodeKey.ACTOR_RNN_STATE in policy_outputs[aid]:
                    rnn_states[aid][EpisodeKey.ACTOR_RNN_STATE] = policy_outputs[aid][EpisodeKey.ACTOR_RNN_STATE]
                if EpisodeKey.CRITIC_RNN_STATE in policy_outputs[aid]:
                    rnn_states[aid][EpisodeKey.CRITIC_RNN_STATE] = policy_outputs[aid][EpisodeKey.CRITIC_RNN_STATE]
                step_data[aid] = d

            if env.is_terminated():
                break

        # compute metrics for this episode
        ep_metrics = compute_episode_metrics(recorder.steps, n_left_ctrl)

        # add existing stats
        stats = env.get_episode_stats()
        if "agent_0" in stats:
            ep_metrics["stats_basic"] = stats["agent_0"]

        # phi diagnostic
        if phi_values:
            ep_metrics["phi_diagnostics"] = {
                "mean": float(np.mean(phi_values)),
                "std": float(np.std(phi_values)),
                "min": float(np.min(phi_values)),
                "max": float(np.max(phi_values)),
            }

        all_episode_metrics.append(ep_metrics)

    env.close()
    return all_episode_metrics


# ---------------------------------------------------------------------------
# Aggregation across episodes
# ---------------------------------------------------------------------------
def aggregate_metrics(episode_metrics: list[dict], n_left_ctrl: int) -> dict:
    """Aggregate per-episode metrics into a summary report."""
    N = len(episode_metrics)
    if N == 0:
        return {}

    results = [ep["result"] for ep in episode_metrics]
    win_rate = sum(1 for r in results if r == "win") / N
    draw_rate = sum(1 for r in results if r == "draw") / N
    loss_rate = sum(1 for r in results if r == "loss") / N

    def _avg(key):
        vals = [ep[key] for ep in episode_metrics if key in ep]
        return float(np.mean(vals)) if vals else 0.0

    def _std(key):
        vals = [ep[key] for ep in episode_metrics if key in ep]
        return float(np.std(vals)) if vals else 0.0

    # territory average
    ter_keys = ["def", "mid", "att"]
    ter_avg = {}
    for k in ter_keys:
        vals = [ep["territory_pct"][k] for ep in episode_metrics if "territory_pct" in ep]
        ter_avg[k] = float(np.mean(vals)) if vals else 0.0

    # per-agent aggregation
    agent_summary = []
    for i in range(n_left_ctrl):
        touches = [ep["agent_touches"][i] for ep in episode_metrics if len(ep.get("agent_touches", [])) > i]
        positions = []
        for ep in episode_metrics:
            for ap in ep.get("agent_avg_positions", []):
                if ap["agent"] == i:
                    positions.append(ap)
        agent_summary.append({
            "agent": i,
            "avg_touches": float(np.mean(touches)) if touches else 0.0,
            "avg_x": float(np.mean([p["avg_x"] for p in positions])) if positions else 0.0,
            "avg_y": float(np.mean([p["avg_y"] for p in positions])) if positions else 0.0,
        })

    # action distribution aggregate
    action_dist_agg = defaultdict(float)
    for ep in episode_metrics:
        for a_name, pct in ep.get("action_distribution", {}).items():
            action_dist_agg[a_name] += pct / N

    # stats_basic aggregate
    basic_keys = [
        "total_pass", "good_pass", "bad_pass",
        "total_shot", "good_shot", "bad_shot",
        "total_possession", "tackle", "get_tackled",
        "interception", "get_intercepted", "total_move",
    ]
    basic_agg = {}
    for k in basic_keys:
        vals = [ep["stats_basic"].get(k, 0) for ep in episode_metrics if "stats_basic" in ep]
        basic_agg[k] = float(np.mean(vals)) if vals else 0.0

    pass_completion = basic_agg["good_pass"] / max(basic_agg["total_pass"], 1)
    shot_accuracy = basic_agg["good_shot"] / max(basic_agg["total_shot"], 1)

    # phi diagnostics aggregate
    phi_diag = None
    phi_eps = [ep["phi_diagnostics"] for ep in episode_metrics if "phi_diagnostics" in ep]
    if phi_eps:
        phi_diag = {
            "mean": float(np.mean([p["mean"] for p in phi_eps])),
            "std": float(np.mean([p["std"] for p in phi_eps])),
            "range": [
                float(np.min([p["min"] for p in phi_eps])),
                float(np.max([p["max"] for p in phi_eps])),
            ],
        }

    # goals aggregation
    goals_for = [ep["final_score_ours"] for ep in episode_metrics]
    goals_against = [ep["final_score_theirs"] for ep in episode_metrics]

    report = {
        "num_episodes": N,
        "outcomes": {
            "win_rate": round(win_rate, 4),
            "draw_rate": round(draw_rate, 4),
            "loss_rate": round(loss_rate, 4),
            "avg_goals_for": round(float(np.mean(goals_for)), 2),
            "avg_goals_against": round(float(np.mean(goals_against)), 2),
        },
        "possession": {
            "avg_rate": round(_avg("possession_rate"), 4),
        },
        "territory": {k: round(v, 4) for k, v in ter_avg.items()},
        "ball_progression": {
            "avg_ball_x": round(_avg("avg_ball_x"), 4),
            "avg_progression_per_step": round(_avg("ball_progression"), 6),
        },
        "formation": {
            "avg_width": round(_avg("avg_team_width"), 4),
            "avg_depth": round(_avg("avg_team_depth"), 4),
            "avg_compactness": round(_avg("avg_compactness"), 4),
        },
        "defense": {
            "avg_line_height": round(_avg("avg_def_line_height"), 4),
            "avg_press_distance": round(_avg("avg_press_distance"), 4),
        },
        "skill_stats": {
            "pass_completion_rate": round(pass_completion, 4),
            "shot_accuracy": round(shot_accuracy, 4),
            "avg_passes_per_game": round(basic_agg["total_pass"], 1),
            "avg_shots_per_game": round(basic_agg["total_shot"], 1),
            "avg_tackles_per_game": round(basic_agg["tackle"], 1),
            "avg_interceptions_per_game": round(basic_agg["interception"], 1),
            "avg_dispossessed_per_game": round(basic_agg["get_tackled"] + basic_agg["get_intercepted"], 1),
        },
        "action_distribution": {k: round(v, 4) for k, v in sorted(action_dist_agg.items())},
        "per_agent": agent_summary,
    }
    if phi_diag:
        report["phi_diagnostics"] = phi_diag

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Extract behavioral metrics from eval episodes")
    ap.add_argument("--checkpoint", required=True, help="Path to agent_0 checkpoint dir")
    ap.add_argument("--config", required=True, help="Training YAML config path")
    ap.add_argument("--opponent", default=None, help="Path to agent_1 checkpoint (default: built-in AI)")
    ap.add_argument("--num-games", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=1, help="Parallel eval workers")
    ap.add_argument("--output", "-o", type=str, default=None, help="Output JSON path")
    ap.add_argument("--phi-module", default=None, help="Python import path for phi (e.g. generated_phi.phi_llm)")
    args = ap.parse_args()

    import yaml
    from omegaconf import OmegaConf

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)

    env_cfg = cfg.rollout_manager.worker.envs[0]
    rollout_length = cfg.rollout_manager.worker.get("eval_rollout_length",
                     cfg.rollout_manager.worker.get("rollout_length", 3001))
    n_left_ctrl = env_cfg.scenario_config.number_of_left_players_agent_controls

    # resolve checkpoint path to actual policy dir
    ckpt_path = resolve_policy_checkpoint(args.checkpoint)
    print(f"Using checkpoint: {ckpt_path}")

    # determine phi module
    phi_mod = args.phi_module
    if phi_mod is None:
        rc = OmegaConf.to_container(env_cfg.get("reward_config", {}), resolve=True)
        ps = rc.get("potential_shaping", {}) if rc else {}
        if ps.get("enabled"):
            phi_mod = ps.get("phi_module")

    # split games across workers
    n_workers = min(args.num_workers, args.num_games)
    games_per_worker = [args.num_games // n_workers] * n_workers
    for i in range(args.num_games % n_workers):
        games_per_worker[i] += 1

    worker_args = [
        (
            ckpt_path,
            args.opponent,
            OmegaConf.to_container(env_cfg, resolve=True),
            rollout_length,
            games_per_worker[w],
            w + 100,
            n_left_ctrl,
            phi_mod,
        )
        for w in range(n_workers)
    ]

    print(f"Running {args.num_games} eval games across {n_workers} worker(s)...")
    t0 = time.time()

    all_episodes = []
    if n_workers == 1:
        all_episodes = _worker_eval(worker_args[0])
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_worker_eval, wa) for wa in worker_args]
            for fut in as_completed(futs):
                all_episodes.extend(fut.result())

    elapsed = time.time() - t0
    print(f"Completed {len(all_episodes)} episodes in {elapsed:.1f}s")

    report = aggregate_metrics(all_episodes, n_left_ctrl)
    report["_meta"] = {
        "checkpoint": ckpt_path,
        "checkpoint_arg": args.checkpoint,
        "config": args.config,
        "num_games": len(all_episodes),
        "phi_module": phi_mod,
        "elapsed_seconds": round(elapsed, 1),
    }

    # output
    out_path = args.output
    if out_path is None:
        ckpt_name = Path(args.checkpoint).name
        out_dir = Path("metrics")
        out_dir.mkdir(exist_ok=True)
        out_path = str(out_dir / f"behavior_{ckpt_name}.json")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=_nan_safe)
    print(f"Report saved to {out_path}")

    # print summary
    print("\n" + "=" * 60)
    print("BEHAVIOR METRICS SUMMARY")
    print("=" * 60)
    o = report["outcomes"]
    print(f"Win/Draw/Loss: {o['win_rate']:.1%} / {o['draw_rate']:.1%} / {o['loss_rate']:.1%}")
    print(f"Goals: {o['avg_goals_for']:.1f} for, {o['avg_goals_against']:.1f} against")
    print(f"Possession: {report['possession']['avg_rate']:.1%}")
    t = report["territory"]
    print(f"Territory: Def {t['def']:.1%} | Mid {t['mid']:.1%} | Att {t['att']:.1%}")
    s = report["skill_stats"]
    print(f"Pass completion: {s['pass_completion_rate']:.1%} ({s['avg_passes_per_game']:.0f}/game)")
    print(f"Shot accuracy: {s['shot_accuracy']:.1%} ({s['avg_shots_per_game']:.0f}/game)")
    fm = report["formation"]
    print(f"Formation: width={fm['avg_width']:.3f} depth={fm['avg_depth']:.3f} compact={fm['avg_compactness']:.3f}")
    d = report["defense"]
    print(f"Defense: line_height={d['avg_line_height']:.3f} press_dist={d['avg_press_distance']:.3f}")
    if "phi_diagnostics" in report:
        pd = report["phi_diagnostics"]
        print(f"Phi: mean={pd['mean']:.3f} std={pd['std']:.3f} range={pd['range']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
