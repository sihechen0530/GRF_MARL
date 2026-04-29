# Copyright 2025 GRF_MARL contributors
"""
Entropy-triggered, regime-gated LLM action masking.

Usage inside policy.compute_action() (inference only):
    # 1. On the previous step's entropy, decide whether to apply a penalty.
    llm_penalty = self.entropy_mask.get_penalty()   # np.ndarray [n_actions]

    # 2. Run actor forward, which applies the penalty to logits (soft, not hard).
    actions, rnn, log_p, entropy = self.actor(..., llm_penalty=llm_penalty)

    # 3. Feed entropy + regime back so the mask can decide whether to call LLM.
    regime = EntropyGuidedMask.extract_regime(observations_np, action_masks_np)
    self.entropy_mask.step(entropy_scalar, regime)

The LLM is called asynchronously in a daemon thread.  While a call is in flight the
mask keeps using whatever penalty it already has (stale-but-useful).  When entropy
drops below the low threshold the penalty is zeroed out regardless of LLM state.

Cache: a plain dict keyed by (game_mode, possession, ball_zone).  With at most
7 × 3 × 3 = 63 unique regimes the dict never needs eviction.
"""
from __future__ import annotations

import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np

from light_malib.utils.logger import Logger
from light_malib.llm.config import LLMClientConfig
from light_malib.llm.openai_compatible import chat_completions_text
from light_malib.llm.prompts_mask import build_mask_messages

# ---------------------------------------------------------------------------
# Constants: encoded obs layout for simple115 + action-mask encoder
# obs shape: [19 action_mask | 22 left_pos | 22 left_dir | 22 right_pos |
#              22 right_dir | 3 ball_xyz | 3 ball_dir | 3 ball_rot |
#              11 active_onehot | 7 game_mode_onehot]
# ---------------------------------------------------------------------------
_N_ACTIONS = 19
_LEFT_POS_START = 19                    # left team x,y positions start here
_BALL_X_IDX = 19 + 22 + 22 + 22 + 22  # = 107
_ACTIVE_START = 19 + 22 + 22 + 22 + 22 + 3 + 3 + 3  # = 116  (active player one-hot)
_GAME_MODE_START = 19 + 22 + 22 + 22 + 22 + 3 + 3 + 3 + 11  # = 127
_SLIDE_IDX = 16   # action index for SLIDE (0 when my team has ball)
_ALWAYS_ZERO_ACTIONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 17)


def _ball_zone(ball_x: float) -> int:
    """Coarse zone: -1 defensive, 0 midfield, 1 attacking (left-team perspective)."""
    if ball_x < -0.3:
        return -1
    if ball_x > 0.3:
        return 1
    return 0


def _parse_llm_response(text: str, n_actions: int) -> Optional[np.ndarray]:
    """
    Parse LLM response into a float32 array of shape [n_actions].
    Accepts a bare JSON array or a JSON array embedded in text.
    Returns None on parse failure.
    """
    text = text.strip()
    # Try to find the first '[' ... ']' block if the model adds extra text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
        if not isinstance(data, list) or len(data) != n_actions:
            Logger.warning(
                f"EntropyGuidedMask: expected {n_actions}-element array, got {len(data) if isinstance(data, list) else type(data)}"
            )
            return None
        arr = np.clip(np.array(data, dtype=np.float32), 0.0, 1.0)
        protected = arr[list(_ALWAYS_ZERO_ACTIONS)]
        if np.any(protected > 0):
            Logger.warning(
                "EntropyGuidedMask: correcting protected-action penalties "
                f"{dict((idx, float(arr[idx])) for idx in _ALWAYS_ZERO_ACTIONS if arr[idx] > 0)}"
            )
            arr[list(_ALWAYS_ZERO_ACTIONS)] = 0.0
        return arr
    except (json.JSONDecodeError, ValueError) as exc:
        Logger.warning(f"EntropyGuidedMask: failed to parse LLM response: {exc}\nResponse: {text[:200]}")
        return None


class EntropyGuidedMask:
    """
    Soft action masking driven by policy entropy and LLM guidance.

    Parameters
    ----------
    n_actions          : action-space size (default 19 for GRF)
    high_thresh        : fraction of max entropy that triggers masking (default 0.7)
    low_thresh         : fraction of max entropy at which masking is disabled (default 0.3)
    penalty_scale      : multiplier applied to LLM weights before subtracting from logits
    num_samples        : number of parallel LLM calls per regime; results are averaged
                         (default 3). More samples = more reliable but more tokens used.
    sample_temperature : LLM temperature for diversity across samples (default 0.4).
                         0.0 would make all samples identical; 0.3-0.5 adds useful variance.
    llm_config         : LLMClientConfig; if None, loaded from environment variables
    """

    def __init__(
        self,
        n_actions: int = _N_ACTIONS,
        high_thresh: float = 0.7,
        low_thresh: float = 0.3,
        penalty_scale: float = 3.0,
        num_samples: int = 3,
        sample_temperature: float = 0.4,
        llm_config: Optional[LLMClientConfig] = None,
    ) -> None:
        self.n_actions = n_actions
        self.max_entropy = math.log(n_actions)
        self.high_thresh = high_thresh * self.max_entropy
        self.low_thresh = low_thresh * self.max_entropy
        self.penalty_scale = penalty_scale
        self.num_samples = num_samples
        self.sample_temperature = sample_temperature

        # Lazy-load LLM config so that EntropyGuidedMask can be instantiated
        # even without an API key (useful for tests); the key is only needed
        # when an actual LLM call is made.
        self._llm_config: Optional[LLMClientConfig] = llm_config

        # Regime cache: (game_mode, possession, ball_zone) -> np.ndarray[n_actions]
        self._cache: dict = {}
        self._lock = threading.Lock()

        # Current state
        self._active: bool = False          # hysteresis state
        self._pending: bool = False         # LLM call in flight
        self._current_regime: Optional[Tuple] = None
        # Weights currently applied (scaled by penalty_scale at apply time)
        self._current_weights: np.ndarray = np.zeros(n_actions, dtype=np.float32)
        self._step_count: int = 0           # total step() calls, for periodic logging

    # ------------------------------------------------------------------
    # Pickle support (threading.Lock is not picklable; Ray serializes
    # the policy object when pushing to remote workers)
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_penalty(self) -> Optional[np.ndarray]:
        """
        Return the current soft penalty array (shape [n_actions], values in
        [0, penalty_scale]), or None if no nonzero penalty is cached.

        Applied regardless of entropy — cached guidance is always used so that
        normal-play (mode=0) regimes, where entropy rarely exceeds the threshold,
        still receive LLM guidance once their cache entry has been populated.

        Call this BEFORE actor.forward() to pass as llm_penalty.
        """
        weights = self._current_weights
        if not np.any(weights > 0):
            return None
        return self.penalty_scale * weights

    def get_penalty_for_regime(self, regime: dict) -> Optional[np.ndarray]:
        """
        Return the cached soft penalty array for a specific regime, or None if the
        regime has not been cached yet or only contains zeros.
        """
        key = self._regime_key(regime)
        with self._lock:
            weights = self._cache.get(key)
            if weights is None:
                return None
            weights = weights.copy()
        if not np.any(weights > 0):
            return None
        return self.penalty_scale * weights

    def step(self, entropy: float, regime: dict) -> None:
        """
        Update internal state with the entropy and regime observed at this step.
        May trigger an async LLM call.  Call AFTER actor.forward().

        Parameters
        ----------
        entropy : scalar entropy of the action distribution at this step
        regime  : dict from extract_regime()
        """
        self._step_count += 1
        if self._step_count % 500 == 1:  # log step 1, 501, 1001, ...
            Logger.info(
                f"EntropyGuidedMask: step={self._step_count} entropy={entropy:.4f} "
                f"high_thresh={self.high_thresh:.4f} low_thresh={self.low_thresh:.4f} "
                f"active={self._active} cache_size={len(self._cache)}"
            )

        # Track high-entropy state for diagnostic logging only (no longer gates cache use)
        if not self._active and entropy > self.high_thresh:
            self._active = True
            Logger.info(
                f"EntropyGuidedMask: high-entropy state (entropy={entropy:.3f} > {self.high_thresh:.3f})"
            )
        elif self._active and entropy < self.low_thresh:
            self._active = False
            Logger.info(
                f"EntropyGuidedMask: low-entropy state (entropy={entropy:.3f} < {self.low_thresh:.3f})"
            )

        # Always look up and populate cache — not gated on entropy.
        # This ensures normal-play (mode 0) regimes, which rarely hit high_thresh,
        # still receive guidance once a cache entry exists.
        key = self._regime_key(regime)

        with self._lock:
            if key in self._cache:
                # Cache hit: update current weights whenever regime changes
                if key != self._current_regime:
                    self._current_weights = self._cache[key]
                    self._current_regime = key
            elif not self._pending:
                # Cache miss and no call in flight: fire async LLM call for any regime
                self._pending = True
                self._current_regime = key
                threading.Thread(
                    target=self._fetch_mask,
                    args=(key, regime),
                    daemon=True,
                ).start()
                Logger.info(
                    f"EntropyGuidedMask: requesting LLM mask for regime {key}"
                )

    @staticmethod
    def extract_regime(obs: np.ndarray, action_mask: np.ndarray) -> dict:
        """
        Derive a compact regime dict from the encoded observation and action mask.

        Works with both single samples (shape [obs_dim]) and batches
        (shape [batch, obs_dim]).  For batches, uses the first element.

        Expects the simple115 + action-mask encoder layout (obs_dim = 134).

        Regime keys:
          game_mode  : int 0-6
          possession : 0=my team, 1=opponent/loose
          ball_zone  : -1=defensive, 0=midfield, 1=attacking
          player_zone: -1=defensive, 0=midfield, 1=attacking (active player's position)
        """
        if obs.ndim > 1:
            obs = obs[0]
            action_mask = action_mask[0]

        # Ball x position (index 107 in 134-dim encoded obs)
        ball_x = float(obs[_BALL_X_IDX]) if len(obs) > _BALL_X_IDX else 0.0

        # Game mode from one-hot (indices 127-133)
        if len(obs) > _GAME_MODE_START + 6:
            game_mode = int(np.argmax(obs[_GAME_MODE_START : _GAME_MODE_START + 7]))
        else:
            game_mode = 0

        # Possession proxy: SLIDE (idx 16) disabled ⟹ my team has ball
        if len(action_mask) > _SLIDE_IDX:
            has_ball = float(action_mask[_SLIDE_IDX]) < 0.5
            possession = 0 if has_ball else 1  # 0=my team, 1=opponent/loose
        else:
            possession = -1

        # Active player zone: one-hot at [116:127], positions interleaved at [19:41]
        player_zone = 0
        if len(obs) > _ACTIVE_START + 10:
            active_idx = int(np.argmax(obs[_ACTIVE_START : _ACTIVE_START + 11]))
            player_x = float(obs[_LEFT_POS_START + active_idx * 2])
            player_zone = _ball_zone(player_x)  # same thresholds as ball zone

        # Compute player_x cleanly for the prompt (already have active_idx above)
        if len(obs) > _ACTIVE_START + 10:
            _ai = int(np.argmax(obs[_ACTIVE_START : _ACTIVE_START + 11]))
            player_x = float(obs[_LEFT_POS_START + _ai * 2])
        else:
            player_x = 0.0

        return {
            "game_mode": game_mode,
            "possession": possession,
            "ball_zone": _ball_zone(ball_x),
            "ball_x": ball_x,
            "player_zone": player_zone,
            "player_x": player_x,
        }

    # ------------------------------------------------------------------
    # Cache persistence
    # ------------------------------------------------------------------

    def save_cache(self, path: str) -> None:
        """
        Persist the regime cache to a JSON file.

        Keys are serialised as "game_mode,possession,ball_zone,player_zone" strings;
        values are lists of floats.  The file is written atomically via a
        temp file so a crash mid-write cannot corrupt the checkpoint.
        """
        import os, tempfile
        with self._lock:
            serialisable = {
                ",".join(str(x) for x in k): v.tolist()
                for k, v in self._cache.items()
            }
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as fh:
                json.dump(serialisable, fh)
            os.replace(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
            raise
        Logger.info(f"EntropyGuidedMask: saved {len(serialisable)} cache entries to {path}")

    def load_cache(self, path: str) -> None:
        """
        Load a previously saved regime cache from *path*.

        Existing in-memory entries are preserved; on-disk entries are merged
        in (disk wins on collision so that fresher checkpoint data is used).
        Missing or malformed files are silently ignored.
        """
        import os
        if not os.path.exists(path):
            return
        try:
            with open(path) as fh:
                raw = json.load(fh)
            loaded: dict = {}
            for key_str, weights_list in raw.items():
                parts = key_str.split(",")
                # Support both old 3-element keys and new 4-element keys.
                # Old keys (game_mode,possession,ball_zone) default player_zone=0.
                if len(parts) == 3:
                    key = (int(parts[0]), int(parts[1]), int(parts[2]), 0)
                else:
                    key = tuple(int(p) for p in parts)
                loaded[key] = np.array(weights_list, dtype=np.float32)
            with self._lock:
                self._cache.update(loaded)
            Logger.info(f"EntropyGuidedMask: loaded {len(loaded)} cache entries from {path}")
        except Exception as exc:
            Logger.warning(f"EntropyGuidedMask: could not load cache from {path}: {exc}")

    def warmup_cache(self) -> None:
        """
        Synchronously pre-populate LLM cache for all normal-play (mode=0) regimes.

        Call this once at training startup (after load_cache) so that the LLM
        guidance is available from the very first training step — before any
        async cache-miss thread has had a chance to return results.

        Covers 18 states: mode=0 × possession∈{0,1} × ball_zone∈{-1,0,1}
                                    × player_zone∈{-1,0,1}
        (possession=-1 is skipped as it's rare in 11v11 normal play).
        Blocking; takes ~10-20 seconds depending on API latency.
        """
        Logger.info("EntropyGuidedMask: warming up cache for normal-play regimes...")
        for ball_zone in (-1, 0, 1):
            for possession in (0, 1):
                for player_zone in (-1, 0, 1):
                    key = (0, possession, ball_zone, player_zone)
                    with self._lock:
                        already_cached = key in self._cache
                    if already_cached:
                        Logger.info(f"EntropyGuidedMask: warmup skipping {key} (already cached)")
                        continue
                    regime = {
                        "game_mode": 0,
                        "possession": possession,
                        "ball_zone": ball_zone,
                        "ball_x": float(ball_zone) * 0.5,
                        "player_zone": player_zone,
                        "player_x": float(player_zone) * 0.5,
                    }
                    Logger.info(f"EntropyGuidedMask: warmup fetching regime {key}")
                    self._fetch_mask(key, regime)   # blocking call
        Logger.info(
            f"EntropyGuidedMask: warmup complete — {len(self._cache)} regimes cached"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _regime_key(regime: dict) -> Tuple:
        return (regime["game_mode"], regime["possession"], regime["ball_zone"],
                regime.get("player_zone", 0))

    def _single_llm_call(self, messages: List[dict]) -> Optional[np.ndarray]:
        """Make one LLM call and parse the response. Returns None on failure."""
        text = chat_completions_text(
            messages,
            config=self._llm_config,
            temperature=self.sample_temperature,
            max_tokens=256,
        )
        return _parse_llm_response(text, self.n_actions)

    def _fetch_mask(self, key: Tuple, regime: dict) -> None:
        """Background thread: fan out num_samples parallel LLM calls and average."""
        try:
            if self._llm_config is None:
                self._llm_config = LLMClientConfig.from_env()
            messages = build_mask_messages(regime)

            samples: List[np.ndarray] = []
            with ThreadPoolExecutor(max_workers=self.num_samples) as pool:
                futures = [pool.submit(self._single_llm_call, messages)
                           for _ in range(self.num_samples)]
                for fut in as_completed(futures):
                    try:
                        result = fut.result()
                        if result is not None:
                            samples.append(result)
                    except Exception as exc:
                        Logger.warning(f"EntropyGuidedMask: sample failed for regime {key}: {exc}")

            if not samples:
                Logger.warning(f"EntropyGuidedMask: all {self.num_samples} samples failed for regime {key}")
                return

            weights = np.mean(samples, axis=0).astype(np.float32)
            with self._lock:
                self._cache[key] = weights
                if self._current_regime == key:
                    self._current_weights = weights

            top = dict(sorted(enumerate(weights), key=lambda x: -x[1])[:3])
            Logger.info(
                f"EntropyGuidedMask: regime {key} averaged {len(samples)}/{self.num_samples} samples, "
                f"top penalties: {top}"
            )
        except Exception as exc:
            Logger.warning(f"EntropyGuidedMask: fetch failed for regime {key}: {exc}")
        finally:
            self._pending = False
