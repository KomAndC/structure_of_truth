"""Noise ceilings via split-half reliability with Spearman-Brown correction."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .seeds import make_rng


def spearman_brown(r: float | np.ndarray, factor: float = 2.0) -> float | np.ndarray:
    """Spearman-Brown prophecy: reliability of a test `factor` times as long."""
    r = np.asarray(r, dtype=float)
    out = factor * r / (1.0 + (factor - 1.0) * r)
    return out.item() if out.ndim == 0 else out


def split_half_ceiling(
    ratings: np.ndarray, *, seed: int, n_splits: int = 200
) -> np.ndarray:
    """Per-concept noise ceiling from repeated random subject splits.

    ratings: (n_subjects, n_concepts, n_concepts) dissimilarity matrices; NaN = missing.
    For each concept i, row i (excluding the diagonal) is averaged within each half of
    subjects, the halves are Spearman-correlated, and the mean split-half r is
    Spearman-Brown corrected to full sample size. Returns shape (n_concepts,).
    """
    ratings = np.asarray(ratings, dtype=float)
    n_sub, n_con, _ = ratings.shape
    if n_sub < 2:
        raise ValueError("need at least 2 subjects for a split-half ceiling")
    rng = make_rng(seed)
    offdiag = ~np.eye(n_con, dtype=bool)
    rs = np.full((n_splits, n_con), np.nan)
    for s in range(n_splits):
        perm = rng.permutation(n_sub)
        a = np.nanmean(ratings[perm[: n_sub // 2]], axis=0)
        b = np.nanmean(ratings[perm[n_sub // 2 :]], axis=0)
        for i in range(n_con):
            x, y = a[i, offdiag[i]], b[i, offdiag[i]]
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() >= 3:
                rs[s, i] = spearmanr(x[ok], y[ok]).statistic
    with np.errstate(invalid="ignore"):
        mean_r = np.nanmean(rs, axis=0)
    return np.asarray(spearman_brown(mean_r))
