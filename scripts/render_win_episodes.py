#!/usr/bin/env python3
"""Render only winning GRF episodes for one trained run/checkpoint."""

import argparse
import csv
import json
import os
import pathlib
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from light_malib.algorithm.mappo.policy import MAPPO
from light_malib.utils.cfg import load_cfg
from light_malib.utils.episode import EpisodeKey
from light_malib.utils.logger import Logger


def discover_checkpoints(run_dir: pathlib.Path, agent_id: str = "agent_0") -> List[Tuple[str, pathlib.Path]]:
    agent_dir = run_dir / agent_id
    if not agent_dir.is_dir():
        return []

    checkpoints = []
    for policy_dir in agent_dir.iterdir():
        if not policy_dir.is_dir():
            continue
        for ckpt_dir in policy_dir.iterdir():
            if ckpt_dir.is_dir() and (ckpt_dir / "desc.pkl").exists():
                checkpoints.append((ckpt_dir.name, ckpt_dir))

    def sort_key(item):
        name = item[0]
        if name == "best":
            return (3, 0)
        match = re.search(r"(\d+)", name)
        epoch = int(match.group(1)) if match else -1
        if name.startswith("epoch_"):
            return (2, epoch)
        if "last" in name:
            return (1, epoch)
        return (0, epoch)

    return sorted(checkpoints, key=sort_key)


def choose_checkpoint(run_dir: pathlib.Path, checkpoint: Optional[str]) -> Tuple[str, pathlib.Path]:
    if checkpoint:
        ckpt_path = pathlib.Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = BASE_DIR / ckpt_path
        return ckpt_path.name, ckpt_path

    eval_csv = run_dir / "eval_results.csv"
    if eval_csv.exists():
        with eval_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            best = max(rows, key=lambda row: float(row.get("win_rate") or 0.0))
            ckpt_name = best["checkpoint"]
            for name, path in discover_checkpoints(run_dir):
                if name == ckpt_name:
                    return name, path
            candidate = run_dir / "agent_0" / "agent_0-default-1" / ckpt_name
            if (candidate / "desc.pkl").exists():
                return ckpt_name, candidate

    checkpoints = discover_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {run_dir}")
    for name, path in reversed(checkpoints):
        if name == "best":
            return name, path
    return checkpoints[-1]


def relpath(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def render_win_episodes(
    cfg,
    checkpoint_dir: pathlib.Path,
    opponent_dir: pathlib.Path,
    output_dir: pathlib.Path,
    num_wins: int,
    max_episodes: int,
    fps: int,
    seed_base: int,
) -> Dict:
    try:
        import cv2
    except ImportError:
        print("[ERROR] opencv-python is required for frame rendering.")
        sys.exit(1)

    from gfootball.env import create_environment as gfootball_create_env
    from light_malib.envs.gr_football.env import register_new_scenarios
    from light_malib.envs.gr_football.state import State

    register_new_scenarios()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_config = OmegaConf.to_container(
        cfg.rollout_manager.worker.envs[0].scenario_config, resolve=True
    )
    scenario_config["render"] = True
    scenario_config["write_video"] = False
    scenario_config["write_full_episode_dumps"] = False

    policy_0 = MAPPO.load(str(checkpoint_dir), env_agent_id="agent_0")
    policy_1 = MAPPO.load(str(opponent_dir), env_agent_id="agent_1")
    policy_0.eval()
    policy_1.eval()

    rollout_length = int(cfg.rollout_manager.worker.eval_rollout_length)
    n_left = int(scenario_config["number_of_left_players_agent_controls"])
    n_right = int(scenario_config["number_of_right_players_agent_controls"])
    use_5v5_state = "5_vs_5" in str(scenario_config.get("env_name", ""))

    manifest_rows = []
    wins = 0

    for attempt in range(max_episodes):
        seed = seed_base + attempt
        other_options = dict(scenario_config.get("other_config_options") or {})
        other_options["game_engine_random_seed"] = int(seed)
        scenario_config["other_config_options"] = other_options

        tmp_video = output_dir / f".attempt_{attempt:04d}.mp4"
        writer = None
        frame_size = None
        raw_env = gfootball_create_env(**scenario_config)
        observations = raw_env.reset()

        rnn_0 = policy_0.get_initial_state(batch_size=n_left)
        rnn_1 = policy_1.get_initial_state(batch_size=n_right)

        done = False
        step = 0
        try:
            while not done and step < rollout_length:
                if len(observations) > 0 and "frame" in observations[0]:
                    frame = observations[0]["frame"]
                    if writer is None:
                        height, width, _ = frame.shape
                        frame_size = (width, height)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(tmp_video), fourcc, fps, frame_size)
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                left_obs = observations[:n_left]
                right_obs = observations[n_left:]
                state_kwargs = {"n_player": 5} if use_5v5_state else {}
                states_0 = [State(**state_kwargs) for _ in range(n_left)]
                states_1 = [State(**state_kwargs) for _ in range(n_right)]
                for obs, state in zip(left_obs, states_0):
                    state.update_obs(obs)
                for obs, state in zip(right_obs, states_1):
                    state.update_obs(obs)

                encoded_0 = np.array(policy_0.feature_encoder.encode(states_0), dtype=np.float32)
                encoded_1 = np.array(policy_1.feature_encoder.encode(states_1), dtype=np.float32)
                action_mask_0 = encoded_0[..., :19]
                action_mask_1 = encoded_1[..., :19]
                done_mask_0 = np.zeros((n_left,), dtype=np.float32)
                done_mask_1 = np.zeros((n_right,), dtype=np.float32)

                out_0 = policy_0.compute_action(
                    inference=True,
                    explore=False,
                    to_numpy=True,
                    step=step,
                    **{
                        EpisodeKey.CUR_OBS: encoded_0,
                        EpisodeKey.ACTION_MASK: action_mask_0,
                        EpisodeKey.DONE: done_mask_0,
                        **rnn_0,
                    },
                )
                out_1 = policy_1.compute_action(
                    inference=True,
                    explore=False,
                    to_numpy=True,
                    step=step,
                    **{
                        EpisodeKey.CUR_OBS: encoded_1,
                        EpisodeKey.ACTION_MASK: action_mask_1,
                        EpisodeKey.DONE: done_mask_1,
                        **rnn_1,
                    },
                )
                rnn_0 = {
                    EpisodeKey.ACTOR_RNN_STATE: out_0[EpisodeKey.ACTOR_RNN_STATE],
                    EpisodeKey.CRITIC_RNN_STATE: out_0[EpisodeKey.CRITIC_RNN_STATE],
                }
                rnn_1 = {
                    EpisodeKey.ACTOR_RNN_STATE: out_1[EpisodeKey.ACTOR_RNN_STATE],
                    EpisodeKey.CRITIC_RNN_STATE: out_1[EpisodeKey.CRITIC_RNN_STATE],
                }
                actions = np.concatenate(
                    [out_0[EpisodeKey.ACTION].flatten(), out_1[EpisodeKey.ACTION].flatten()]
                )
                observations, _, done, _ = raw_env.step(actions.astype(np.int32))
                step += 1
        finally:
            if writer is not None:
                writer.release()
            raw_env.close()

        my_score = int(observations[0]["score"][0]) if len(observations) else 0
        opp_score = int(observations[0]["score"][1]) if len(observations) else 0
        outcome = "win" if my_score > opp_score else ("loss" if my_score < opp_score else "draw")
        saved_path = ""

        if outcome == "win" and tmp_video.exists() and frame_size is not None:
            saved_path_obj = output_dir / f"win_{wins:03d}_seed{seed}_{my_score}-{opp_score}.mp4"
            tmp_video.rename(saved_path_obj)
            saved_path = relpath(saved_path_obj)
            wins += 1
            Logger.info(f"Saved win {wins}/{num_wins}: {saved_path}")
        elif tmp_video.exists():
            tmp_video.unlink()

        manifest_rows.append(
            {
                "attempt": attempt,
                "seed": seed,
                "outcome": outcome,
                "score": f"{my_score}-{opp_score}",
                "steps": step,
                "saved_video": saved_path,
            }
        )
        Logger.info(
            f"Attempt {attempt + 1}/{max_episodes}: {outcome.upper()} "
            f"{my_score}-{opp_score}; wins saved {wins}/{num_wins}"
        )
        if wins >= num_wins:
            break

    manifest_csv = output_dir / "manifest.csv"
    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["attempt", "seed", "outcome", "score", "steps", "saved_video"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "checkpoint": relpath(checkpoint_dir),
        "opponent": relpath(opponent_dir),
        "output_dir": relpath(output_dir),
        "requested_wins": num_wins,
        "saved_wins": wins,
        "attempts": len(manifest_rows),
        "max_episodes": max_episodes,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-wins", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20260419)
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path
    run_dir = pathlib.Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = BASE_DIR / run_dir
    opponent_dir = pathlib.Path(args.opponent)
    if not opponent_dir.is_absolute():
        opponent_dir = BASE_DIR / opponent_dir
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir

    ckpt_name, checkpoint_dir = choose_checkpoint(run_dir, args.checkpoint)
    if not (checkpoint_dir / "desc.pkl").exists():
        raise FileNotFoundError(f"Checkpoint is missing desc.pkl: {checkpoint_dir}")
    if not (opponent_dir / "desc.pkl").exists():
        raise FileNotFoundError(f"Opponent is missing desc.pkl: {opponent_dir}")

    Logger.info(f"Config: {config_path}")
    Logger.info(f"Run: {run_dir}")
    Logger.info(f"Checkpoint: {ckpt_name} -> {checkpoint_dir}")
    Logger.info(f"Opponent: {opponent_dir}")
    Logger.info(f"Output: {output_dir}")

    cfg = load_cfg(str(config_path))
    summary = render_win_episodes(
        cfg=cfg,
        checkpoint_dir=checkpoint_dir,
        opponent_dir=opponent_dir,
        output_dir=output_dir,
        num_wins=args.num_wins,
        max_episodes=args.max_episodes,
        fps=args.fps,
        seed_base=args.seed_base,
    )
    Logger.info(f"Summary: {summary}")


if __name__ == "__main__":
    main()
