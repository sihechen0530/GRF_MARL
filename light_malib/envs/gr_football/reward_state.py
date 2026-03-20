# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
# Licensed under the Apache License, Version 2.0 (the "License");
import numpy as np


def _as_xy(pos):
    a = np.asarray(pos, dtype=np.float64).reshape(-1)
    return float(a[0]), float(a[1]) if a.size > 1 else 0.0


def _team_matrix(obs, team_key: str, dir_key: str, roles_key: str):
    """GRF lists all on-pitch players per team; length follows roles, not controlled-agent count."""
    if roles_key not in obs:
        raise ValueError(f"obs missing {roles_key}")
    n = int(np.asarray(obs[roles_key], dtype=np.int64).size)
    if n <= 0:
        raise ValueError(f"empty {roles_key}")
    flat = np.asarray(obs[team_key], dtype=np.float64).reshape(-1)
    if flat.size % n != 0:
        raise ValueError(
            f"cannot reshape {team_key}: n_roles={n}, flat.size={flat.size}"
        )
    d = flat.size // n
    xy = flat.reshape(n, d)
    dir_flat = np.asarray(obs.get(dir_key, np.zeros(flat.size)), dtype=np.float64).reshape(-1)
    if dir_flat.size == flat.size:
        direction = dir_flat.reshape(n, d)
    else:
        direction = np.zeros_like(xy)
    return xy, direction, n


def extract_reward_state(obs, player_idx: int, n_left: int, n_right: int) -> dict:
    """
    Compact dict for LLM-authored or hand-written Φ(state, role).
    Coordinates follow GRF raw layout: x in [-1, 1], y in [-0.42, 0.42] typically.
    Left team attacks toward +x; right team toward -x.
    """
    if obs is None:
        raise ValueError("obs is None")

    n_total = n_left + n_right
    if not (0 <= player_idx < n_total):
        raise ValueError(f"player_idx {player_idx} out of range for n_total={n_total}")

    left_xy, left_dir, n_lp = _team_matrix(
        obs, "left_team", "left_team_direction", "left_team_roles"
    )
    right_xy, right_dir, n_rp = _team_matrix(
        obs, "right_team", "right_team_direction", "right_team_roles"
    )

    ball = np.asarray(obs["ball"], dtype=np.float64).reshape(-1)
    ball_pos = (float(ball[0]), float(ball[1]), float(ball[2]) if ball.size > 2 else 0.0)
    ball_dir = np.asarray(obs.get("ball_direction", np.zeros(3)), dtype=np.float64).reshape(-1)
    ball_vel = (
        float(ball_dir[0]) if ball_dir.size > 0 else 0.0,
        float(ball_dir[1]) if ball_dir.size > 1 else 0.0,
        float(ball_dir[2]) if ball_dir.size > 2 else 0.0,
    )

    is_left = player_idx < n_left
    if is_left:
        li = player_idx
        if li >= n_lp:
            raise ValueError(
                f"left player_idx {player_idx} >= n_left_players_in_obs={n_lp}"
            )
        my_pos = _as_xy(left_xy[li, :2])
        my_vel = _as_xy(left_dir[li, :2]) if left_dir.shape[0] > li else (0.0, 0.0)
        attack_goal = np.array([1.0, 0.0], dtype=np.float64)
        teammates_xy = [tuple(_as_xy(left_xy[j, :2])) for j in range(n_lp) if j != li]
        opponents_xy = [tuple(_as_xy(right_xy[j, :2])) for j in range(n_rp)]
    else:
        ri = player_idx - n_left
        if ri >= n_rp:
            raise ValueError(
                f"right player_idx {player_idx} maps to ri={ri} >= n_right_in_obs={n_rp}"
            )
        my_pos = _as_xy(right_xy[ri, :2])
        my_vel = _as_xy(right_dir[ri, :2]) if right_dir.shape[0] > ri else (0.0, 0.0)
        attack_goal = np.array([-1.0, 0.0], dtype=np.float64)
        teammates_xy = [tuple(_as_xy(right_xy[j, :2])) for j in range(n_rp) if j != ri]
        opponents_xy = [tuple(_as_xy(left_xy[j, :2])) for j in range(n_lp)]

    bot = int(obs["ball_owned_team"])
    bop = int(obs["ball_owned_player"])
    active = obs.get("active")
    if active is not None:
        active_i = int(np.asarray(active, dtype=np.int64).reshape(-1)[0])
    else:
        active_i = li if is_left else ri
    if is_left:
        has_ball = bot == 0 and bop >= 0 and int(bop) == active_i
    else:
        has_ball = bot == 1 and bop >= 0 and int(bop) == active_i

    my_xy = np.array(my_pos, dtype=np.float64)
    ball_xy = np.array(ball_pos[:2], dtype=np.float64)
    dist_to_goal = float(np.linalg.norm(attack_goal - my_xy))
    ball_dist_to_goal = float(np.linalg.norm(attack_goal - ball_xy))
    dist_to_ball = float(np.linalg.norm(ball_xy - my_xy))

    if len(opponents_xy) == 0:
        nearest_opp_dist = 1.0
    else:
        opp = np.asarray(opponents_xy, dtype=np.float64)
        nearest_opp_dist = float(np.min(np.linalg.norm(opp - my_xy[None, :], axis=1)))

    score = obs["score"]
    if is_left:
        score_diff = int(score[0]) - int(score[1])
    else:
        score_diff = int(score[1]) - int(score[0])

    return {
        "player_idx": int(player_idx),
        "is_left_team": bool(is_left),
        "my_pos": my_pos,
        "my_vel": my_vel,
        "has_ball": bool(has_ball),
        "ball_pos": ball_pos,
        "ball_vel": ball_vel,
        "dist_to_goal": dist_to_goal,
        "ball_dist_to_goal": ball_dist_to_goal,
        "dist_to_ball": dist_to_ball,
        "nearest_opp_dist": nearest_opp_dist,
        "teammates_pos": teammates_xy,
        "opponents_pos": opponents_xy,
        "game_mode": int(obs["game_mode"]),
        "score_diff": score_diff,
        "steps_left": int(obs["steps_left"]),
    }
