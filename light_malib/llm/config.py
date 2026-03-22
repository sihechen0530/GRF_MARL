# Copyright 2025 GRF_MARL contributors
"""Environment-driven settings for OpenAI-compatible chat APIs (DeepSeek, OpenAI, vLLM, etc.)."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMClientConfig:
    """Minimal config for POST /v1/chat/completions style endpoints."""

    base_url: str
    api_key: str
    model: str
    timeout_s: float = 120.0

    @classmethod
    def from_env(cls) -> LLMClientConfig:
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing API key: set LLM_API_KEY (preferred) or DEEPSEEK_API_KEY"
            )
        base = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
        model = os.environ.get("LLM_MODEL", "deepseek-chat")
        timeout_s = float(os.environ.get("LLM_TIMEOUT_S", "120"))
        return cls(base_url=base, api_key=api_key, model=model, timeout_s=timeout_s)
