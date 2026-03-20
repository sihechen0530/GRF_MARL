# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
# Licensed under the Apache License, Version 2.0 (the "License");
"""
Hand-written potential Φ for Eureka-style PoC (replace with LLM-generated module later).

Contract:
  - Φ(state, role) -> float in [0, 1]
  - Training uses F = γ Φ(s') - Φ(s) per step; combined reward = r_env + alpha * F
"""


def phi(state: dict, role: str) -> float:
    ball_progress = 1.0 - min(state["ball_dist_to_goal"] / 1.5, 1.0)
    my_prog = 1.0 - min(state["dist_to_goal"] / 1.5, 1.0)

    if role == "ball_carrier":
        p = 0.55 * ball_progress + 0.35 * my_prog
        if state["has_ball"] and state["ball_pos"][0] > 0.5:
            p += 0.1
    elif role in ("left_winger", "right_winger", "support_wide"):
        ahead = max(0.0, state["my_pos"][0] - state["ball_pos"][0])
        width = abs(state["my_pos"][1])
        p = (
            0.35 * ball_progress
            + 0.25 * min(ahead / 0.5, 1.0)
            + 0.25 * min(width / 0.3, 1.0)
            + 0.15 * (1.0 - min(state["dist_to_ball"] / 0.8, 1.0))
        )
    elif role in ("trailing_mid", "support_trail", "midfield_support"):
        behind = max(0.0, state["ball_pos"][0] - state["my_pos"][0])
        p = (
            0.4 * ball_progress
            + 0.3 * (1.0 - min(state["dist_to_ball"] / 0.8, 1.0))
            + 0.3 * min(behind / 0.3, 1.0)
        )
    else:
        p = 0.55 * ball_progress + 0.25 * my_prog

    return float(max(0.0, min(1.0, p)))
