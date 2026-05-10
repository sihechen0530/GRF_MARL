"""
Environment wrapper that computes LLM action-penalty vectors and injects them
into each step's observation dict as EpisodeKey.LLM_PENALTY.

The penalty travels through the episode buffer alongside the observation, so the
exact same distribution used during rollout is replayed during PPO training —
eliminating the rollout/training importance-ratio mismatch caused by the old
policy-side EntropyGuidedMask stateful approach.

Data flow
---------
  env.step()  →  { agent_id: { NEXT_OBS: ..., LLM_PENALTY: penalty, ... } }
                                              ↑
                 stored verbatim in episode buffer

  loss.py     →  reads LLM_PENALTY from batch → passes to compute_action(inference=False)
                 → new_log_prob computed under same distribution as old_log_prob  ✓
"""
from __future__ import annotations

import numpy as np

from light_malib.utils.episode import EpisodeKey
from light_malib.utils.logger import Logger
from light_malib.llm.entropy_mask import EntropyGuidedMask


class GRFLLMMaskWrapper:
    """
    Wraps a GRFootballEnv to inject LLM action penalties into every step's data.

    Parameters
    ----------
    env            : GRFootballEnv instance to wrap.
    custom_config  : Policy custom_config dict; reads llm_mask_* keys.
    training_agent : Agent ID that receives computed penalties (opponent gets zeros).
    """

    def __init__(self, env, custom_config: dict, training_agent: str = "agent_0"):
        self._env = env
        self._training_agent = training_agent
        self.warmup_epochs = custom_config.get("llm_mask_warmup_epochs", 500)
        self.current_epoch = 0
        self._eval = False

        self.entropy_mask = EntropyGuidedMask(
            n_actions=env.num_actions,
            high_thresh=custom_config.get("llm_mask_high_thresh", 0.7),
            low_thresh=custom_config.get("llm_mask_low_thresh", 0.3),
            penalty_scale=custom_config.get("llm_mask_penalty_scale", 0.3),
            num_samples=custom_config.get("llm_mask_num_samples", 3),
            sample_temperature=custom_config.get("llm_mask_sample_temperature", 0.4),
        )

        if custom_config.get("llm_mask_warmup", False):
            self.entropy_mask.warmup_cache()

        Logger.info(
            f"GRFLLMMaskWrapper: initialized (warmup_epochs={self.warmup_epochs}, "
            f"penalty_scale={custom_config.get('llm_mask_penalty_scale', 0.3)})"
        )

    # ------------------------------------------------------------------
    # Attribute delegation to inner env
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        # Only called when the attribute is NOT found on the wrapper itself.
        # Guards against recursion if _env isn't set yet.
        if name == "_env":
            raise AttributeError(name)
        return getattr(self._env, name)

    # ------------------------------------------------------------------
    # Overridden env interface
    # ------------------------------------------------------------------

    def reset(self, custom_reset_config: dict) -> dict:
        self.current_epoch = custom_reset_config.get("rollout_epoch", 0)
        self._eval = custom_reset_config.get("eval", False)
        env_rets = self._env.reset(custom_reset_config)
        self._inject_penalty(env_rets)
        return env_rets

    def step(self, actions: dict) -> dict:
        env_rets = self._env.step(actions)
        self._inject_penalty(env_rets)
        return env_rets

    def is_terminated(self) -> bool:
        return self._env.is_terminated()

    def get_episode_stats(self) -> dict:
        return self._env.get_episode_stats()

    def get_AssistInfo(self) -> dict:
        return self._env.get_AssistInfo()

    def render(self, mode: str = "human"):
        return self._env.render(mode)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inject_penalty(self, env_rets: dict) -> None:
        """Add EpisodeKey.LLM_PENALTY to every agent's observation dict in-place."""
        for agent_id, agent_rets in env_rets.items():
            obs = agent_rets[EpisodeKey.NEXT_OBS]             # [num_players, obs_dim]
            action_mask = agent_rets[EpisodeKey.ACTION_MASK]  # [num_players, n_actions]
            n_players = obs.shape[0]

            if agent_id == self._training_agent and not self._eval:
                penalties = self._compute_penalties(obs, action_mask)
            else:
                penalties = None

            if penalties is not None:
                agent_rets[EpisodeKey.LLM_PENALTY] = penalties
            else:
                agent_rets[EpisodeKey.LLM_PENALTY] = np.zeros(
                    (n_players, self._env.num_actions), dtype=np.float32
                )

    def _compute_penalties(
        self, obs: np.ndarray, action_mask: np.ndarray
    ):
        """
        Update the entropy mask for each controlled player and return a penalty matrix
        of shape [num_players, n_actions], or None during the warmup period.

        entropy=1.0 is a placeholder; the mask no longer gates on entropy — it
        calls the LLM for every unseen regime regardless.
        """
        if self.current_epoch < self.warmup_epochs:
            return None

        penalties = np.zeros((obs.shape[0], self._env.num_actions), dtype=np.float32)
        for player_idx in range(obs.shape[0]):
            regime = EntropyGuidedMask.extract_regime(
                obs[player_idx], action_mask[player_idx]
            )
            self.entropy_mask.step(1.0, regime)
            penalty = self.entropy_mask.get_penalty_for_regime(regime)
            if penalty is not None:
                penalties[player_idx] = penalty
        return penalties
