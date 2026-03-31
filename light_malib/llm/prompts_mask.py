# Copyright 2025 GRF_MARL contributors
"""
Prompt templates for LLM-guided action masking.

The LLM is asked to return a soft penalty weight in [0, 1] per action given
a compact game regime summary.  The penalty is applied as:
    logits -= penalty_scale * llm_weights
so 0.0 = no guidance, 1.0 = strong discouragement (not a hard block).
"""
from __future__ import annotations

from typing import Dict, List

# GRF action index → human-readable name (19-action space)
ACTION_NAMES: Dict[int, str] = {
    0: "no_op",
    1: "left",
    2: "top_left",
    3: "top",
    4: "top_right",
    5: "right",
    6: "bottom_right",
    7: "bottom",
    8: "bottom_left",
    9: "long_pass",
    10: "high_pass",
    11: "short_pass",
    12: "shot",
    13: "sprint",
    14: "release_move",
    15: "release_sprint",
    16: "slide",
    17: "dribble",
    18: "release_dribble",
}

GAME_MODE_NAMES: Dict[int, str] = {
    0: "Normal",
    1: "KickOff",
    2: "GoalKick",
    3: "FreeKick",
    4: "Corner",
    5: "ThrowIn",
    6: "Penalty",
}

MASK_SYSTEM_PROMPT = """\
You are an expert in Google Research Football (GRF) multi-agent reinforcement learning.

Your task: given a brief description of the current game REGIME, return a JSON array of
exactly 19 floats representing SOFT PENALTY WEIGHTS for each action.

Penalty weight semantics:
  0.0 = no penalty (action is fine or neutral in this regime)
  1.0 = strong penalty (action is strategically bad in this regime)
  Values in-between are allowed for partial discouragement.

The 19 GRF actions (index → name):
  0:no_op  1:left  2:top_left  3:top  4:top_right  5:right  6:bottom_right
  7:bottom  8:bottom_left  9:long_pass  10:high_pass  11:short_pass  12:shot
  13:sprint  14:release_move  15:release_sprint  16:slide  17:dribble  18:release_dribble

Rules:
1. Output ONLY a JSON array of exactly 19 floats. No explanation, no markdown, no code block.
2. Respect hard legality: the environment already blocks truly illegal moves; your weights
   guide STRATEGIC choices on top of that.
3. Keep ALL penalties at 0.0 by default. Only assign a non-zero penalty when you are HIGHLY
   CONFIDENT the action is counter-productive. Most entries should be 0.0.
4. Actions 1-8 (directional movement) must always be 0.0.
5. Action 12 (shot) must ALWAYS be 0.0. The agent must be free to shoot at any time;
   whether a shot is physically possible is handled by the environment.
6. Action 13 (sprint) must ALWAYS be 0.0.
7. Action 16 (slide tackle) should only receive a small penalty (max 0.3) in specific
   attacking situations where sliding is clearly wrong (e.g. my team has the ball deep in
   the attacking third). Do NOT penalize slide in defensive situations.
8. Penalties above 0.3 are almost never appropriate — the RL agent needs freedom to explore.
"""


def _possession_str(possession: int) -> str:
    return {0: "my team", 1: "opponent", -1: "loose / contested"}[possession]


def _zone_str(zone: int) -> str:
    return {-1: "defensive third (x < -0.3)", 0: "midfield (-0.3 ≤ x ≤ 0.3)", 1: "attacking third (x > 0.3)"}[zone]


def build_mask_user_prompt(regime: dict) -> str:
    """
    regime keys:
      game_mode   : int 0-6
      possession  : int  0=my team, 1=opponent, -1=loose
      ball_zone   : int -1=defensive, 0=midfield, 1=attacking
      ball_x      : float (raw, for extra context)
    """
    mode_name = GAME_MODE_NAMES.get(regime["game_mode"], f"Unknown({regime['game_mode']})")
    poss_str = _possession_str(regime["possession"])
    zone_str = _zone_str(regime["ball_zone"])
    ball_x = regime.get("ball_x", 0.0)
    return (
        f"Game regime:\n"
        f"  Game mode   : {mode_name} (id={regime['game_mode']})\n"
        f"  Possession  : {poss_str}\n"
        f"  Ball zone   : {zone_str} (ball_x ≈ {ball_x:.2f})\n\n"
        f"Return the 19-element JSON penalty array now."
    )


def build_mask_messages(regime: dict) -> List[dict]:
    """Return OpenAI-format messages list for the action masking request."""
    return [
        {"role": "system", "content": MASK_SYSTEM_PROMPT},
        {"role": "user", "content": build_mask_user_prompt(regime)},
    ]
