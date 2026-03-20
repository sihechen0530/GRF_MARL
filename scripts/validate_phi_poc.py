#!/usr/bin/env python3
"""Sanity-check a phi module (bounded, non-degenerate on synthetic states)."""
import argparse
import sys
from pathlib import Path

# Running as `python3 scripts/validate_phi_poc.py` puts scripts/ on sys.path, not repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from light_malib.envs.gr_football.potential_shaping import load_phi_callable, validate_phi


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phi-module",
        type=str,
        default="light_malib.envs.gr_football.phi_poc",
        help="Import path; module must define phi(state: dict, role: str) -> float",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    phi = load_phi_callable(args.phi_module)
    validate_phi(phi, np.random.default_rng(args.seed), n_left=4, n_right=2, n_samples=128)
    print("validate_phi: OK")


if __name__ == "__main__":
    main()
