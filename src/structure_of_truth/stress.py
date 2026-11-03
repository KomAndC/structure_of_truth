"""Kruskal stress-1, decomposed per concept-pair and per concept.

Given observed dissimilarities ``δ_ij`` and configuration distances ``d_ij`` (both n x n,
symmetric, only i < j used), stress-1 is::

    S = sqrt( sum_{i<j} (dhat_ij - d_ij)^2 / sum_{i<j} d_ij^2 )

where ``dhat`` are the *disparities* — the best transformation of ``δ`` allowed by the MDS
level:

* ``"raw"``      dhat = δ (no transformation; scale-dependent);
* ``"ratio"``    dhat = b δ, b by least squares (default; invariant to scaling δ or d);
* ``"interval"`` dhat = a + b δ, least squares;
* ``"ordinal"``  dhat = isotonic (monotone non-decreasing) regression of d on δ.

Decomposition (sums to S^2, so contributions are directly comparable):

* ``pair[i, j] = (dhat_ij - d_ij)^2 / sum_{i<j} d_ij^2`` (symmetric, zero diagonal);
  ``pair[iu].sum() == S^2``.
* ``concept[i] = 0.5 * sum_j pair[i, j]`` — concept i's share of the misfit;
  ``concept.sum() == S^2``. ``concept_stress1[i]`` is the row-normalised stress-1 analogue
  ``sqrt(sum_j (dhat_ij - d_ij)^2 / sum_j d_ij^2)``, useful for comparing concepts with
  different typical distances.

NaN in ``δ`` excludes that pair from every sum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.isotonic import IsotonicRegression


@dataclass
class StressResult:
    stress1: float
    pair: np.ndarray  # (n, n)
    concept: np.ndarray  # (n,)
    concept_stress1: np.ndarray  # (n,)
    disparities: np.ndarray  # (n, n), NaN where δ missing
    level: str


def _disparities(delta: np.ndarray, d: np.ndarray, level: str) -> np.ndarray:
    if level == "raw":
        return delta
    if level == "ratio":
        den = (delta**2).sum()
        return delta * ((delta * d).sum() / den) if den > 0 else np.zeros_like(delta)
    if level == "interval":
        Xd = np.column_stack([np.ones_like(delta), delta])
        coef, *_ = np.linalg.lstsq(Xd, d, rcond=None)
        return Xd @ coef
    if level == "ordinal":
        return IsotonicRegression(increasing=True).fit_transform(delta, d)
    raise ValueError("level must be 'raw', 'ratio', 'interval' or 'ordinal'")


def stress(delta: np.ndarray, distances: np.ndarray, *, level: str = "ratio") -> StressResult:
    """Kruskal stress-1 of fitted distances against observed dissimilarities."""
    delta = np.asarray(delta, dtype=float)
    D = np.asarray(distances, dtype=float)
    n = delta.shape[0]
    if delta.shape != (n, n) or D.shape != (n, n):
        raise ValueError("delta and distances must both be n x n")
    iu = np.triu_indices(n, k=1)
    dv, Dv = delta[iu], D[iu]
    ok = ~np.isnan(dv)
    denom = float((Dv[ok] ** 2).sum())
    if denom == 0:
        raise ValueError("all fitted distances are zero")
    dhat = np.full_like(dv, np.nan)
    dhat[ok] = _disparities(dv[ok], Dv[ok], level)
    e = np.where(ok, (dhat - Dv) ** 2, 0.0)

    pair = np.zeros((n, n))
    pair[iu] = e / denom
    pair = pair + pair.T
    concept = 0.5 * pair.sum(axis=1)

    sq = np.zeros((n, n))
    sq[iu] = e
    sq = sq + sq.T
    dd = np.zeros((n, n))
    dd[iu] = np.where(ok, Dv**2, 0.0)
    dd = dd + dd.T
    with np.errstate(invalid="ignore", divide="ignore"):
        cs1 = np.sqrt(sq.sum(axis=1) / dd.sum(axis=1))

    disp = np.full((n, n), np.nan)
    disp[iu] = dhat
    disp.T[iu] = dhat
    return StressResult(float(np.sqrt(e.sum() / denom)), pair, concept, cs1, disp, level)


def stress_from_config(delta: np.ndarray, X: np.ndarray, *, level: str = "ratio") -> StressResult:
    """Stress of Euclidean distances of configuration ``X`` (n x dim) against ``delta``."""
    return stress(delta, squareform(pdist(np.asarray(X, dtype=float))), level=level)
