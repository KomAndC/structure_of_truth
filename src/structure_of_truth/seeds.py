"""Explicit-seed randomness. Never use np.random.* global state or unseeded RNGs."""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return a numpy Generator for an explicit integer seed."""
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    return np.random.default_rng(int(seed))


def child_seeds(seed: int, n: int) -> list[int]:
    """Derive n independent, reproducible integer seeds from a parent seed."""
    ss = np.random.SeedSequence(int(seed))
    return [int(c.generate_state(1)[0]) for c in ss.spawn(n)]
