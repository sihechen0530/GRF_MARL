# Copyright 2025 GRF_MARL contributors
"""Extract Python source from LLM replies (markdown fences or bare module)."""
from __future__ import annotations

import re
from typing import Optional


_FENCE_PY = re.compile(
    r"```(?:python|py)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_FENCE_ANY = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def extract_python_code_block(text: str) -> str:
    """
    Prefer a ```python ... ``` block; otherwise first generic ``` ... ``` block.
    If no fence, return stripped text if it looks like a module (contains 'def phi').
    """
    if not text or not str(text).strip():
        raise ValueError("empty model output")

    t = str(text).strip()
    m = _FENCE_PY.search(t)
    if m:
        return m.group(1).strip()
    m = _FENCE_ANY.search(t)
    if m:
        return m.group(1).strip()
    if "def phi" in t:
        return t
    raise ValueError(
        "Could not find ```python ... ``` block and output does not contain def phi"
    )


def ensure_phi_export(source: str) -> str:
    """Light sanity check; raises if 'def phi' missing."""
    if "def phi" not in source:
        raise ValueError("extracted code must define def phi(state, role) -> float")
    return source
