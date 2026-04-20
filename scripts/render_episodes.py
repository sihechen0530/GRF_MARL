#!/usr/bin/env python3
"""
Render evaluation episodes into videos.

Loads a trained checkpoint, plays games against the built-in AI, and saves
videos and/or dump files. GRF's built-in renderer writes MP4 videos when
`write_video=True`.

On headless machines (e.g. SLURM), wrap with xvfb:
    xvfb-run -a python scripts/render_episodes.py ...

Usage:
    # Record 5 episodes as MP4 videos
    python scripts/render_episodes.py \
        --config expr_configs/cooperative_MARL_benchmark/academy/3_vs_1_with_keeper/mappo.yaml \
        --checkpoint logs/.../agent_0/default-1/best \
        --num_episodes 5 --output_dir videos/3v1_mappo

    # Record with GRF dump files (for replay in the visual debugger)
    python scripts/render_episodes.py \
        --config expr_configs/cooperative_MARL_benchmark/academy/3_vs_1_with_keeper/mappo.yaml \
        --checkpoint logs/.../agent_0/default-1/best \
        --num_episodes 3 --save_dumps --output_dir videos/3v1_mappo

    # Scan and render all epoch checkpoints from a run, 1 episode each
    python scripts/render_episodes.py \
        --config expr_configs/cooperative_MARL_benchmark/academy/3_vs_1_with_keeper/mappo.yaml \
        --run_dir logs/gr_football/benchmark_academy_3_vs_1_with_keeper_mappo/<timestamp> \
        --num_episodes 1 --output_dir videos/3v1_mappo_progression

    # Assemble a video from raw RGB frames (works without display)
    python scripts/render_episodes.py \
        --config expr_configs/cooperative_MARL_benchmark/academy/3_vs_1_with_keeper/mappo.yaml \
        --checkpoint logs/.../agent_0/default-1/best \
        --num_episodes 3 --mode frames --output_dir videos/3v1_mappo

    # Only save winning episodes (roll until 2 wins, cap 120 tries)
    python scripts/render_episodes.py ... \
        --num_episodes 2 --wins-only --max-attempts 120 --mode frames --output_dir videos/wins
"""

import argparse
import os
import sys
import re
import copy
import shutil
import pathlib
from collections import OrderedDict

import numpy as np
from omegaconf import OmegaConf

BASE_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, BASE_DIR)

from light_malib.algorithm.mappo.policy import MAPPO
from light_malib.utils.cfg import load_cfg
from light_malib.utils.logger import Logger


def discover_checkpoints(run_dir, agent_id="agent_0", epoch_interval=None):
    """Find all checkpoint directories under a run directory."""
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
            if not os.path.exists(os.path.join(ckpt_dir, "desc.pkl")):
                continue
            checkpoints.append((ckpt_name, ckpt_dir))

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
            if m and int(m.group(1)) % epoch_interval != 0:
                continue
            filtered.append((name, path))
        checkpoints = filtered

    return checkpoints


def render_with_gfr_video(cfg, checkpoint_dir, opponent_dir, num_episodes, output_dir, save_dumps):
    """
    Use GRF's built-in video writer.
    Requires a display (real or xvfb).
    """
    from light_malib.envs.gr_football.env import GRFootballEnv
    from light_malib.rollout.rollout_func import rollout_func
    from light_malib.utils.desc.task_desc import RolloutDesc

    os.makedirs(output_dir, exist_ok=True)
    env_cfg = copy.deepcopy(cfg.rollout_manager.worker.envs[0])
    OmegaConf.set_struct(env_cfg, False)
    env_cfg.scenario_config.render = True
    env_cfg.scenario_config.write_video = True
    env_cfg.scenario_config.write_full_episode_dumps = save_dumps
    env_cfg.scenario_config.logdir = output_dir

    policy_0 = MAPPO.load(checkpoint_dir, env_agent_id="agent_0")
    policy_1 = MAPPO.load(opponent_dir, env_agent_id="agent_1")

    rollout_length = cfg.rollout_manager.worker.eval_rollout_length
    rollout_desc = RolloutDesc("agent_0", None, None, None, None, None)
    behavior_policies = {
        "agent_0": ("policy_0", policy_0),
        "agent_1": ("policy_1", policy_1),
    }

    wins = []
    for ep in range(num_episodes):
        env = GRFootballEnv(ep, None, env_cfg)
        Logger.info(f"Recording episode {ep + 1}/{num_episodes}...")

        rollout_results = rollout_func(
            eval=True,
            rollout_worker=None,
            rollout_desc=rollout_desc,
            env=env,
            behavior_policies=behavior_policies,
            data_server=None,
            rollout_length=rollout_length,
            render=True,
            rollout_epoch=100,
        )

        for result in rollout_results["results"]:
            stats = result["stats"]["agent_0"]
            outcome = "WIN" if stats["win"] == 1 else ("LOSS" if stats["lose"] == 1 else "DRAW")
            wins.append(stats["win"])
            Logger.info(f"  Episode {ep + 1}: {outcome} (score: {stats['my_goal']}, reward: {stats['reward']:.2f})")

        env._env.close()

    Logger.info(f"Win rate: {np.mean(wins):.1%} ({sum(int(w) for w in wins)}/{len(wins)})")
    try:
        contents = os.listdir(output_dir)
        if not any(f.endswith((".mp4", ".avi", ".dump")) for f in contents):
            Logger.warning(
                f"No video/dump files in {output_dir} (contents: {contents}). "
                "Try --mode frames to capture video via OpenCV."
            )
    except OSError:
        pass
    Logger.info(f"Videos saved to: {output_dir}")


def render_with_frames(
    cfg,
    checkpoint_dir,
    opponent_dir,
    num_episodes,
    output_dir,
    fps=10,
    wins_only=False,
    max_attempts=80,
):
    """
    Capture RGB frames from GRF observations and assemble into MP4.
    Requires `render=True` in env config so frames appear in observations,
    but uses opencv to write the video file.

    If wins_only: roll until num_episodes wins are saved as win_000.mp4, ...
    (non-win attempts are not written). max_attempts caps total tries.
    """
    try:
        import cv2
    except ImportError:
        print("[ERROR] opencv-python is required for frame mode. Install with: pip install opencv-python")
        sys.exit(1)

    from gfootball.env import create_environment as gfootball_create_env
    from light_malib.envs.gr_football.env import register_new_scenarios

    register_new_scenarios()

    os.makedirs(output_dir, exist_ok=True)

    scenario_config = dict(cfg.rollout_manager.worker.envs[0].scenario_config)
    scenario_config["render"] = True
    scenario_config["write_video"] = False
    scenario_config["write_full_episode_dumps"] = False

    policy_0 = MAPPO.load(checkpoint_dir, env_agent_id="agent_0")
    policy_1 = MAPPO.load(opponent_dir, env_agent_id="agent_1")
    policy_0.eval()
    policy_1.eval()

    from light_malib.utils.episode import EpisodeKey

    rollout_length = cfg.rollout_manager.worker.eval_rollout_length
    n_left = scenario_config["number_of_left_players_agent_controls"]
    n_right = scenario_config["number_of_right_players_agent_controls"]

    wins = []
    saved_wins = 0
    attempt = 0
    while True:
        attempt += 1
        if wins_only:
            if saved_wins >= num_episodes:
                break
            if attempt > max_attempts:
                Logger.error(
                    f"wins-only: got {saved_wins} win(s) after {max_attempts} attempts "
                    f"(need {num_episodes}). Raise --max-attempts or pick a stronger checkpoint."
                )
                sys.exit(1)
            Logger.info(
                f"Attempt {attempt}/{max_attempts} (saving win {saved_wins + 1}/{num_episodes})..."
            )
        else:
            if attempt > num_episodes:
                break
            Logger.info(f"Recording episode {attempt}/{num_episodes}...")

        raw_env = gfootball_create_env(**scenario_config)
        observations = raw_env.reset()

        frames = []
        if len(observations) > 0 and "frame" in observations[0]:
            frames.append(observations[0]["frame"].copy())

        done = False
        step = 0
        while not done and step < rollout_length:
            left_obs = observations[:n_left]
            right_obs = observations[n_left:]

            fe_0 = policy_0.feature_encoder
            fe_1 = policy_1.feature_encoder

            from light_malib.envs.gr_football.state import State
            states_0 = [State() for _ in range(n_left)]
            states_1 = [State() for _ in range(n_right)]
            for o, s in zip(left_obs, states_0):
                s.update_obs(o)
            for o, s in zip(right_obs, states_1):
                s.update_obs(o)

            encoded_0 = np.array(fe_0.encode(states_0), dtype=np.float32)
            encoded_1 = np.array(fe_1.encode(states_1), dtype=np.float32)

            action_mask_0 = encoded_0[..., :19]
            action_mask_1 = encoded_1[..., :19]

            init_rnn_0 = policy_0.get_initial_state(batch_size=n_left)
            init_rnn_1 = policy_1.get_initial_state(batch_size=n_right)
            # RNN mask: 0 = not done, 1 = done (required by MAPPO compute_action)
            done_mask_0 = np.zeros((n_left,), dtype=np.float32)
            done_mask_1 = np.zeros((n_right,), dtype=np.float32)

            out_0 = policy_0.compute_action(
                inference=True, explore=False, to_numpy=True, step=0,
                **{
                    EpisodeKey.CUR_OBS: encoded_0,
                    EpisodeKey.ACTION_MASK: action_mask_0,
                    EpisodeKey.DONE: done_mask_0,
                    **init_rnn_0,
                }
            )
            out_1 = policy_1.compute_action(
                inference=True, explore=False, to_numpy=True, step=0,
                **{
                    EpisodeKey.CUR_OBS: encoded_1,
                    EpisodeKey.ACTION_MASK: action_mask_1,
                    EpisodeKey.DONE: done_mask_1,
                    **init_rnn_1,
                }
            )

            actions = np.concatenate([
                out_0[EpisodeKey.ACTION].flatten(),
                out_1[EpisodeKey.ACTION].flatten(),
            ])

            observations, rewards, done, info = raw_env.step(actions.astype(np.int32))

            if len(observations) > 0 and "frame" in observations[0]:
                frames.append(observations[0]["frame"].copy())

            step += 1

        my_score = observations[0]["score"][0] if len(observations) > 0 else 0
        opp_score = observations[0]["score"][1] if len(observations) > 0 else 0
        win = 1 if my_score > opp_score else 0
        outcome = "WIN" if my_score > opp_score else ("LOSS" if my_score < opp_score else "DRAW")
        wins.append(win)
        Logger.info(f"  Result: {outcome} ({my_score}-{opp_score}, {len(frames)} frames)")

        raw_env.close()

        if wins_only and not win:
            Logger.info("  (not a win — discarding, no file written)")
            continue

        if frames:
            if wins_only:
                video_path = os.path.join(output_dir, f"win_{saved_wins:03d}.mp4")
            else:
                video_path = os.path.join(
                    output_dir, f"episode_{attempt - 1:03d}_{outcome.lower()}.mp4"
                )
            h, w, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            Logger.info(f"  Saved: {video_path}")
            if wins_only:
                saved_wins += 1
        else:
            Logger.warning(
                "  No frames captured for this attempt. "
                "Make sure 'render=True' is supported by your GRF build."
            )

    if wins:
        Logger.info(f"\nWin rate over attempts: {np.mean(wins):.1%} ({sum(wins)}/{len(wins)})")
    if wins_only:
        Logger.info(f"wins-only: saved {saved_wins} video(s) under {output_dir}")
    Logger.info(f"Videos saved to: {output_dir}")


def _collect_native_videos(episode_logdir, output_dir, ep_index):
    """If GRF wrote video/dump into episode_logdir (or cwd), copy to output_dir with a clear name."""
    import glob
    import shutil
    import time
    copied = []
    max_age_sec = 120  # when searching cwd, only take files modified in last 2 min

    def collect_from_dir(search_dir, require_recent=False):
        found = []
        for ext in ("*.mp4", "*.avi", "*.dump"):
            for path in glob.glob(os.path.join(search_dir, ext)):
                if require_recent and (time.time() - os.path.getmtime(path)) > max_age_sec:
                    continue
                found.append(path)
        return found

    # Search in the requested logdir first
    if os.path.isdir(episode_logdir):
        paths = collect_from_dir(episode_logdir)
    else:
        paths = []
    # gfootball sometimes writes to cwd; check there only if logdir had nothing (recent files only)
    if not paths:
        paths = collect_from_dir(os.getcwd(), require_recent=True)
    for i, path in enumerate(paths):
        suffix = path.split(".")[-1]
        out_name = f"episode_{ep_index:03d}.{suffix}" if i == 0 else f"episode_{ep_index:03d}_{i}.{suffix}"
        out_path = os.path.join(output_dir, out_name)
        try:
            shutil.copy2(path, out_path)
            copied.append(out_path)
        except Exception as e:
            Logger.warning(f"Could not copy {path} to {out_path}: {e}")
    return copied


def render_with_rollout_func(
    cfg,
    checkpoint_dir,
    opponent_dir,
    num_episodes,
    output_dir,
    save_dumps,
    wins_only=False,
    max_attempts=80,
):
    """
    Use the project's own rollout_func with GRF's native video writer.
    Simpler and more reliable than frame capture, but requires a display.

    wins_only: keep rolling until num_episodes wins are collected; copy only winning
    episode videos as win_000.mp4, ... Non-win attempt dirs are removed.
    """
    from light_malib.envs.gr_football.env import GRFootballEnv
    from light_malib.rollout.rollout_func import rollout_func
    from light_malib.utils.desc.task_desc import RolloutDesc

    os.makedirs(output_dir, exist_ok=True)

    policy_0 = MAPPO.load(checkpoint_dir, env_agent_id="agent_0")
    policy_1 = MAPPO.load(opponent_dir, env_agent_id="agent_1")

    rollout_length = cfg.rollout_manager.worker.eval_rollout_length
    rollout_desc = RolloutDesc("agent_0", None, None, None, None, None)
    behavior_policies = {
        "agent_0": ("policy_0", policy_0),
        "agent_1": ("policy_1", policy_1),
    }

    wins = []
    saved_wins = 0
    attempt = 0

    while True:
        attempt += 1
        if wins_only:
            if saved_wins >= num_episodes:
                break
            if attempt > max_attempts:
                Logger.error(
                    f"wins-only: got {saved_wins} win(s) after {max_attempts} attempts "
                    f"(need {num_episodes}). Raise --max-attempts or pick a stronger checkpoint."
                )
                sys.exit(1)
            Logger.info(
                f"Attempt {attempt}/{max_attempts} (saving win {saved_wins + 1}/{num_episodes})..."
            )
            ep_tag = f"attempt_{attempt:04d}"
        else:
            if attempt > num_episodes:
                break
            Logger.info(f"Recording episode {attempt}/{num_episodes}...")
            ep_tag = f"episode_{attempt - 1:03d}"

        episode_logdir = os.path.abspath(os.path.join(output_dir, ep_tag))
        os.makedirs(episode_logdir, exist_ok=True)

        env_cfg = copy.deepcopy(cfg.rollout_manager.worker.envs[0])
        OmegaConf.set_struct(env_cfg, False)
        env_cfg.scenario_config.render = True
        env_cfg.scenario_config.write_video = True
        env_cfg.scenario_config.write_full_episode_dumps = save_dumps
        env_cfg.scenario_config.logdir = episode_logdir

        env = GRFootballEnv(attempt, None, env_cfg)

        rollout_results = rollout_func(
            eval=True,
            rollout_worker=None,
            rollout_desc=rollout_desc,
            env=env,
            behavior_policies=behavior_policies,
            data_server=None,
            rollout_length=rollout_length,
            render=True,
            rollout_epoch=100,
        )

        win_val = 0
        for result in rollout_results["results"]:
            stats = result["stats"]["agent_0"]
            outcome = "WIN" if stats["win"] == 1 else ("LOSS" if stats["lose"] == 1 else "DRAW")
            win_val = int(stats["win"])
            wins.append(win_val)
            Logger.info(
                f"  Result: {outcome} (goals: {stats['my_goal']}, reward: {stats['reward']:.2f})"
            )

        env._env.close()

        # Unique scratch index for wins_only so episode_*.mp4 in output_dir never collide
        scratch_idx = (attempt - 1) if not wins_only else (500 + attempt)
        copied = _collect_native_videos(episode_logdir, output_dir, scratch_idx)

        if wins_only and not win_val:
            Logger.info("  (not a win — discarding scratch video, removing attempt dir)")
            for p in copied:
                try:
                    if os.path.isfile(p):
                        os.remove(p)
                except OSError:
                    pass
            shutil.rmtree(episode_logdir, ignore_errors=True)
            continue

        if wins_only and win_val:
            mp4s = [p for p in copied if p.endswith(".mp4")]
            if mp4s:
                dst = os.path.join(output_dir, f"win_{saved_wins:03d}.mp4")
                try:
                    shutil.move(mp4s[0], dst)
                    Logger.info(f"  Saved: {dst}")
                except OSError as e:
                    Logger.warning(f"  Could not move {mp4s[0]} -> {dst}: {e}")
                for extra in mp4s[1:]:
                    try:
                        os.remove(extra)
                    except OSError:
                        pass
                saved_wins += 1
            else:
                Logger.warning("  Win but no mp4 collected; try --mode frames")
            shutil.rmtree(episode_logdir, ignore_errors=True)
            continue

        if copied:
            Logger.info(f"  Native output: {copied}")
        else:
            try:
                contents = os.listdir(episode_logdir)
                Logger.warning(
                    f"  No video/dump files in {episode_logdir} (contents: {contents}). "
                    "Try --mode frames to capture video via OpenCV."
                )
            except OSError:
                Logger.warning(
                    f"  No video/dump files produced for attempt {attempt}. "
                    "Try --mode frames to capture video via OpenCV."
                )

    if wins:
        Logger.info(f"\nWin rate over attempts: {np.mean(wins):.1%} ({sum(int(w) for w in wins)}/{len(wins)})")
    if wins_only:
        Logger.info(f"wins-only: saved {saved_wins} win clip(s) under {output_dir}")
    Logger.info(f"Videos saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Render evaluation episodes into videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  native   Use GRF's built-in video writer (requires display or xvfb)
  frames   Capture RGB frames and assemble with OpenCV (requires display or xvfb + opencv-python)

On headless machines, wrap the command with xvfb:
  xvfb-run -a python scripts/render_episodes.py ...
        """,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment config YAML")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a single checkpoint directory")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Path to run directory (renders all/filtered checkpoints)")
    parser.add_argument("--opponent", type=str,
                        default="light_malib/trained_models/gr_football/5_vs_5/built_in",
                        help="Path to opponent model")
    parser.add_argument("--num_episodes", type=int, default=3,
                        help="Number of episodes to record per checkpoint (default: 3)")
    parser.add_argument("--output_dir", type=str, default="videos",
                        help="Output directory for videos (default: videos/)")
    parser.add_argument("--mode", type=str, default="native", choices=["native", "frames"],
                        help="Rendering mode (default: native)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Video FPS for frames mode (default: 10)")
    parser.add_argument("--save_dumps", action="store_true",
                        help="Also save GRF dump files (for visual debugger replay)")
    parser.add_argument("--agent_id", type=str, default="agent_0")
    parser.add_argument("--epoch_interval", type=int, default=None,
                        help="Only render every Nth epoch checkpoint (when using --run_dir)")
    parser.add_argument("--filter", nargs="*", default=None,
                        help="Filter checkpoint names (e.g. best last epoch_500)")
    parser.add_argument(
        "--wins-only",
        action="store_true",
        help="Keep sampling until --num-episodes wins are saved (win_000.mp4, ...). "
        "Non-wins are not written (frames mode) or scratch files removed (native).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=80,
        help="With --wins-only, max environment resets before giving up (default: 80).",
    )
    args = parser.parse_args()

    if args.checkpoint is None and args.run_dir is None:
        print("[ERROR] Must specify either --checkpoint or --run_dir")
        sys.exit(1)

    config_path = os.path.join(BASE_DIR, args.config)
    cfg = load_cfg(config_path)
    opponent_dir = os.path.join(BASE_DIR, args.opponent)

    if args.checkpoint:
        checkpoints = [("custom", args.checkpoint)]
    else:
        checkpoints = discover_checkpoints(args.run_dir, args.agent_id, args.epoch_interval)
        if args.filter:
            checkpoints = [
                (name, path) for name, path in checkpoints
                if any(f in name for f in args.filter)
            ]

    if not checkpoints:
        print("[ERROR] No checkpoints found")
        sys.exit(1)

    if args.wins_only:
        print(
            f"Will save {args.num_episodes} WIN video(s) per checkpoint "
            f"(wins-only, max {args.max_attempts} attempts each)"
        )
    else:
        print(f"Will render {args.num_episodes} episode(s) for {len(checkpoints)} checkpoint(s)")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output_dir}\n")

    for name, ckpt_dir in checkpoints:
        print(f"\n{'='*60}")
        print(f"Checkpoint: {name}")
        print(f"  Path: {ckpt_dir}")
        print(f"{'='*60}")

        ckpt_output = os.path.join(args.output_dir, name)
        os.makedirs(ckpt_output, exist_ok=True)

        if args.mode == "native":
            render_with_rollout_func(
                cfg,
                ckpt_dir,
                opponent_dir,
                args.num_episodes,
                ckpt_output,
                args.save_dumps,
                wins_only=args.wins_only,
                max_attempts=args.max_attempts,
            )
        elif args.mode == "frames":
            render_with_frames(
                cfg,
                ckpt_dir,
                opponent_dir,
                args.num_episodes,
                ckpt_output,
                args.fps,
                wins_only=args.wins_only,
                max_attempts=args.max_attempts,
            )

    print(f"\nAll videos saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
