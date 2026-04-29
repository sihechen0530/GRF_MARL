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

Your task: given a description of the current game situation, return a JSON array of
exactly 19 floats representing SOFT PENALTY WEIGHTS for each action.

Penalty weight semantics:
  0.0 = no penalty (action is fine or neutral)
  1.0 = strong penalty (action is strategically bad in this situation)

The 19 GRF actions (index → name):
  0:no_op  1:left  2:top_left  3:top  4:top_right  5:right  6:bottom_right
  7:bottom  8:bottom_left  9:long_pass  10:high_pass  11:short_pass  12:shot
  13:sprint  14:release_move  15:release_sprint  16:slide  17:dribble  18:release_dribble

Hard rules (NEVER violate):
1. Output ONLY a JSON array of exactly 19 floats. No explanation, no markdown.
2. Actions 1-8 (directional movement): ALWAYS 0.0.
3. Action 9 (long_pass): ALWAYS 0.0. Never penalize long passes.
4. Action 10 (high_pass): ALWAYS 0.0. Never penalize high passes.
5. Action 12 (shot): ALWAYS 0.0. Never penalize shooting.
6. Action 13 (sprint): ALWAYS 0.0. Never penalize sprinting.
7. Action 17 (dribble): ALWAYS 0.0. Never penalize dribbling.

Guidance by situation — use these as strong defaults:

WHEN OPPONENT HAS BALL (possession=opponent):
  - Action 18 (release_dribble): 0.5
  - Action 0 (no_op): 0.5 in defensive zone, 0.3 elsewhere  [passivity is punished]
  - Action 16 (slide): 0.0  [sliding tackle is appropriate defensive action]

WHEN MY TEAM HAS BALL (possession=my_team):
  - Action 16 (slide): 0.7 in attacking zone, 0.3 in midfield  [don't slide when attacking]
  - Action 0 (no_op): 0.5  [passivity wastes possession]

PLAYER IN DEFENSIVE ZONE (player_zone=defensive):
  - If opponent has ball: Action 0 (no_op): 0.6  [must press or position]
  - If my team has ball: Action 16 (slide): 0.1  [mild discourage; focus on passing out]

PLAYER IN ATTACKING ZONE (player_zone=attacking):
  - If my team has ball: Action 16 (slide): 0.8, Action 0 (no_op): 0.4
  - If opponent has ball: Action 0 (no_op): 0.6  [must press or recover defensively]

SET PIECE MODES (game_mode != 0):
  - KickOff (mode 1): Action 16 (slide): 0.8
  - GoalKick (mode 2), FreeKick (mode 3), Corner (mode 4):
      If my team has ball: Action 0 (no_op): 0.6, Action 16 (slide): 0.9
      If opponent has ball: Action 0 (no_op): 0.5
  - Penalty (mode 6): If my team has ball: 0.8 for actions 0,11,16,18  [shoot, don't stall]

The RL agent needs freedom to explore. Apply penalties confidently where shown above,
but do not invent penalties for situations not covered — leave those at 0.0.
"""


def _possession_str(possession: int) -> str:
    return {0: "my team", 1: "opponent", -1: "loose / contested"}[possession]


def _zone_str(zone: int) -> str:
    return {
        -1: "defensive third (x < -0.3)",
        0: "midfield (-0.3 ≤ x ≤ 0.3)",
        1: "attacking third (x > 0.3)",
    }[zone]


def build_mask_user_prompt(regime: dict) -> str:
    """
    regime keys:
      game_mode   : int 0-6
      possession  : int  0=my team, 1=opponent, -1=loose
      ball_zone   : int -1=defensive, 0=midfield, 1=attacking
      ball_x      : float (raw, for extra context)
      player_zone : int -1=defensive, 0=midfield, 1=attacking (active player position)
      player_x    : float (raw active player x)
    """
    mode_name = GAME_MODE_NAMES.get(regime["game_mode"], f"Unknown({regime['game_mode']})")
    poss_str = _possession_str(regime["possession"])
    ball_zone_str = _zone_str(regime["ball_zone"])
    player_zone = regime.get("player_zone", 0)
    player_zone_str = _zone_str(player_zone)
    ball_x = regime.get("ball_x", 0.0)
    player_x = regime.get("player_x", 0.0)
    return (
        f"Game situation:\n"
        f"  Game mode   : {mode_name} (id={regime['game_mode']})\n"
        f"  Possession  : {poss_str}\n"
        f"  Ball zone   : {ball_zone_str} (ball_x ≈ {ball_x:.2f})\n"
        f"  Player zone : {player_zone_str} (player_x ≈ {player_x:.2f})\n\n"
        f"Return the 19-element JSON penalty array now."
    )


def build_mask_messages(regime: dict) -> List[dict]:
    """Return OpenAI-format messages list for the action masking request."""
    return [
        {"role": "system", "content": MASK_SYSTEM_PROMPT},
        {"role": "user", "content": build_mask_user_prompt(regime)},
    ]
