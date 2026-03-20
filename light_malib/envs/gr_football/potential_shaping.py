# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
# Licensed under the Apache License, Version 2.0 (the "License");
from __future__ import annotations

import importlib
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from light_malib.envs.gr_football.reward_state import extract_reward_state


def load_phi_callable(phi_module: str) -> Callable[[dict, str], float]:
    mod = importlib.import_module(phi_module)
    if not hasattr(mod, "phi"):
        raise AttributeError(f"Module {phi_module} must define phi(state, role) -> float")
    return getattr(mod, "phi")


def resolve_role(
    player_idx: int,
    n_left: int,
    n_right: int,
    obs,
    left_roles: Sequence[str],
    right_roles: Optional[Sequence[str]],
    right_default: str,
) -> str:
    bot = int(obs["ball_owned_team"])
    bop = int(obs["ball_owned_player"])

    if player_idx < n_left:
        li = player_idx
        active_local = int(np.asarray(obs.get("active", li)).reshape(-1)[0])
        if bot == 0 and bop == active_local:
            return "ball_carrier"
        if not left_roles:
            return "default"
        return str(left_roles[player_idx % len(left_roles)])

    ri = player_idx - n_left
    active_local = int(np.asarray(obs.get("active", ri)).reshape(-1)[0])
    if bot == 1 and bop == active_local:
        return "ball_carrier"
    roles = right_roles if right_roles else []
    if not roles:
        return right_default
    return str(roles[ri % len(roles)])


def validate_phi(
    phi_fn: Callable[[dict, str], float],
    rng: np.random.Generator,
    n_left: int = 4,
    n_right: int = 2,
    n_samples: int = 64,
    roles: Optional[List[str]] = None,
) -> None:
    roles = roles or ["ball_carrier", "left_winger", "right_winger", "trailing_mid", "default"]
    vals = []
    for _ in range(n_samples):
        obs = _synthetic_obs(rng, n_left, n_right)
        for pi in range(n_left + n_right):
            st = extract_reward_state(obs, pi, n_left, n_right)
            for r in roles:
                v = float(phi_fn(st, r))
                if not isinstance(v, (float, int)) or np.isnan(v) or np.isinf(v):
                    raise ValueError(f"phi returned non-finite value: {v}")
                if v < 0.0 or v > 1.0:
                    raise ValueError(f"phi out of [0,1]: {v} for role={r}")
                vals.append(v)
    spread = max(vals) - min(vals)
    if spread < 1e-3:
        raise ValueError("phi appears nearly constant on synthetic states (degenerate)")


def _synthetic_obs(rng: np.random.Generator, n_left: int, n_right: int) -> dict:
    left = rng.uniform(-1, 1, size=(n_left, 2)).astype(np.float32)
    right = rng.uniform(-1, 1, size=(n_right, 2)).astype(np.float32)
    ball = rng.uniform(-1, 1, size=(3,)).astype(np.float32)
    return {
        "left_team": left,
        "right_team": right,
        "left_team_direction": rng.normal(size=(n_left, 2)).astype(np.float32) * 0.01,
        "right_team_direction": rng.normal(size=(n_right, 2)).astype(np.float32) * 0.01,
        "ball": ball,
        "ball_direction": rng.normal(size=(3,)).astype(np.float32) * 0.01,
        "ball_owned_team": int(rng.integers(-1, 2)),
        "ball_owned_player": int(rng.integers(0, max(n_left, n_right))),
        "game_mode": int(rng.integers(0, 7)),
        "score": [int(rng.integers(0, 3)), int(rng.integers(0, 3))],
        "steps_left": int(rng.integers(0, 400)),
        "left_team_roles": np.zeros(n_left, dtype=np.int32),
        "right_team_roles": np.zeros(n_right, dtype=np.int32),
        "active": 0,
    }
