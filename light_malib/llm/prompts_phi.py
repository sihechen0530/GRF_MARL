# Copyright 2025 GRF_MARL contributors
"""
Prompt templates for Eureka-style potential functions Φ(state, role).

Kept in one module so experiments can version or swap templates without touching HTTP.
"""
from __future__ import annotations

from typing import List, Tuple

# --- System: role + hard constraints (high-signal, implementation-focused) ---

PHI_SYSTEM_PROMPT = """You are an expert in reinforcement learning and multi-agent Google Research Football (GRF).

Your job is to write a single Python module that defines ONE function:

    def phi(state: dict, role: str) -> float:

This function is a potential Φ used in potential-based shaping:
    F = γ * Φ(s') - Φ(s)
Training adds α * F to the environment reward (α, γ are handled outside your code).

Requirements (must all hold):
1) Pure function: no I/O, no randomness, no globals, no reading files or environment.
2) Use only the Python standard library and `numpy` imported as `np` if needed.
3) Return a finite float in [0.0, 1.0] for every valid input.
4) `state` is a dict with the keys and types below (values may vary by scenario).
5) Branch on `role` to encode role-specific priorities (e.g. ball_carrier vs winger).
6) Prefer smooth, bounded combinations of distances (normalize by typical GRF scales: field x roughly [-1,1], y roughly [-0.42,0.42]).
7) Output ONLY a single ```python code block``` containing the full module source (imports + phi). No explanation outside the code block.

State dict contract (from extract_reward_state):
- player_idx: int
- is_left_team: bool
- my_pos: tuple of two floats (x, y)
- my_vel: tuple of two floats
- has_ball: bool
- ball_pos: tuple of three floats (x, y, z) — use x,y for ground geometry
- ball_vel: tuple of three floats
- dist_to_goal: float — distance from this agent to its attack goal line in plane
- ball_dist_to_goal: float — ball to that same attack goal
- dist_to_ball: float
- nearest_opp_dist: float — distance to closest opponent (defaults to 1.0 if none)
- teammates_pos: list of (x,y) tuples for teammates on pitch (excluding self)
- opponents_pos: list of (x,y) tuples
- game_mode: int
- score_diff: int — goal difference from this agent's team perspective
- steps_left: int

Role strings are decided by training config; you MUST handle at least:
- "ball_carrier"
- "default"
and any additional roles may appear — map unknown roles to sensible behavior (e.g. treat like "default").

Implementation hints:
- Encourage progressing the ball toward the opponent goal when it helps the team; penalize absurd positioning via bounded terms.
- For non-carriers, reward spacing/support relative to ball without hardcoding illegal play.
- Clip final output with float(max(0.0, min(1.0, x))) if needed.
"""


def build_phi_user_prompt(
    *,
    extra_task_hint: str = "",
    left_roles_example: str = "ball_carrier, left_winger, right_winger, trailing_mid, default",
    scenario_name: str = "GRF academy / full-game MARL",
) -> str:
    """User message: scenario + optional team-specific hints."""
    hint = extra_task_hint.strip()
    hint_block = (
        f"\nAdditional task-specific instructions from the human:\n{hint}\n"
        if hint
        else ""
    )
    return f"""Generate `phi(state, role)` for cooperative MARL in {scenario_name}.

Typical left-team role labels (examples, not exhaustive): {left_roles_example}
{hint_block}
Remember: one ```python``` block only, full module with imports if any, defining `phi`.
"""


def build_phi_messages(
    *,
    extra_task_hint: str = "",
    left_roles_example: str = "ball_carrier, left_winger, right_winger, trailing_mid, default",
    scenario_name: str = "GRF academy / full-game MARL",
) -> List[Tuple[str, str]]:
    """Return (role, content) pairs for chat_completions `messages` list format."""
    return [
        ("system", PHI_SYSTEM_PROMPT),
        (
            "user",
            build_phi_user_prompt(
                extra_task_hint=extra_task_hint,
                left_roles_example=left_roles_example,
                scenario_name=scenario_name,
            ),
        ),
    ]


def messages_as_openai_json(
    pairs: List[Tuple[str, str]],
) -> List[dict]:
    return [{"role": r, "content": c} for r, c in pairs]


# ---------------------------------------------------------------------------
# Iterative Eureka: diagnose behavior metrics & revise phi
# ---------------------------------------------------------------------------

ITERATIVE_EUREKA_SYSTEM_PROMPT = """You are an expert in reinforcement learning reward shaping for multi-agent Google Research Football (GRF).

You are given:
1) The CURRENT potential function `phi(state, role) -> float` used for reward shaping.
2) BEHAVIORAL METRICS collected from evaluation episodes of agents trained with this phi.
3) Optionally, a DIAGNOSIS REQUEST from the human specifying what to improve.

Your task:
A) DIAGNOSE: Analyze the behavioral metrics to identify weaknesses or suboptimal behaviors. Provide a concise analysis (3-5 bullet points).
B) REVISE: Write an improved `phi(state, role)` that addresses the diagnosed issues.

The revised phi MUST satisfy all the same constraints as the original:
- Pure function: no I/O, no randomness, no globals.
- Use only stdlib + numpy.
- Return float in [0.0, 1.0].
- Branch on role string.
- State dict keys: player_idx, is_left_team, my_pos, my_vel, has_ball, ball_pos, ball_vel,
  dist_to_goal, ball_dist_to_goal, dist_to_ball, nearest_opp_dist, teammates_pos,
  opponents_pos, game_mode, score_diff, steps_left.

Output format (STRICT):
1) First output your diagnosis inside a ```text block```.
2) Then output the revised phi in a single ```python block```.
No other text outside these two blocks.
"""


def build_iterative_eureka_messages(
    *,
    current_phi_code: str,
    behavior_metrics_json: str,
    iteration: int = 1,
    diagnosis_request: str = "",
    scenario_name: str = "GRF 11v11 full game",
) -> List[Tuple[str, str]]:
    """Build chat messages for iterative Eureka phi revision."""
    user_parts = [
        f"## Iteration {iteration} — Revise phi for {scenario_name}\n",
        "### Current phi code:\n```python\n" + current_phi_code.strip() + "\n```\n",
        "### Behavioral metrics from evaluation:\n```json\n" + behavior_metrics_json.strip() + "\n```\n",
    ]
    if diagnosis_request.strip():
        user_parts.append(
            f"### Human diagnosis request:\n{diagnosis_request.strip()}\n"
        )
    user_parts.append(
        "\nDiagnose the issues and output an improved phi. "
        "Remember: ```text``` block for diagnosis, then ```python``` block for revised phi."
    )
    return [
        ("system", ITERATIVE_EUREKA_SYSTEM_PROMPT),
        ("user", "\n".join(user_parts)),
    ]
