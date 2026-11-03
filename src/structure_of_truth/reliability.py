"""Noise ceilings: per-concept and dataset-level split-half reliability.

Input everywhere: ``ratings`` of shape (n_subjects, n_concepts, n_concepts), one symmetric
dissimilarity matrix per subject; NaN marks a missing judgment (ignored by ``nanmean``).

Per concept i (``split_half_per_concept``)
    For each of ``n_splits`` (default 200) seeded random splits of subjects into two
    halves (sizes floor(K/2) and ceil(K/2)), average each half's matrices, take row i
    without the diagonal as that half's *dissimilarity profile* of concept i, correlate
    the two profiles (Spearman by default), average r over splits, and apply the
    Spearman–Brown correction with factor 2 to estimate the reliability of the full-sample
    profile. Profiles with < 3 jointly observed entries give NaN for that split.

Dataset level (``dataset_noise_ceiling``)
    * ``split_half``: the same procedure on the full upper triangle of the half-mean
      matrices, Spearman–Brown corrected.
    * ``lower`` / ``upper`` (Nili et al. 2014): mean over subjects of the correlation of
      the subject's upper triangle with the mean of the *other* subjects (lower bound) or
      of *all* subjects including itself (upper bound). A model of the group structure
      cannot be expected to beat the upper bound.

Spearman–Brown is applied to the mean r, not averaged per split; negative mean r gives a
negative (uninterpretable) ceiling and is returned as is rather than clipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

from .seeds import make_rng


def spearman_brown(r: float | np.ndarray, factor: float = 2.0) -> float | np.ndarray:
    """Spearman–Brown prophecy: reliability of a measure ``factor`` times as long."""
    r = np.asarray(r, dtype=float)
    out = factor * r / (1.0 + (factor - 1.0) * r)
    return out.item() if out.ndim == 0 else out


def _check(ratings: np.ndarray) -> np.ndarray:
    ratings = np.asarray(ratings, dtype=float)
    if ratings.ndim != 3 or ratings.shape[1] != ratings.shape[2]:
        raise ValueError("ratings must have shape (n_subjects, n_concepts, n_concepts)")
    if ratings.shape[0] < 2:
        raise ValueError("need at least 2 subjects")
    return ratings


def _corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan
    f = spearmanr if method == "spearman" else pearsonr
    return float(f(x[ok], y[ok]).statistic)


def _rowwise_corr(a: np.ndarray, b: np.ndarray, method: str) -> np.ndarray:
    """Correlate matching rows of two complete (no-NaN) 2-D arrays."""
    if method == "spearman":
        a, b = rankdata(a, axis=1), rankdata(b, axis=1)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(axis=1) / den, np.nan)


def _halves(rng: np.random.Generator, K: int) -> tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(K)
    return perm[: K // 2], perm[K // 2 :]


@dataclass
class ConceptReliability:
    ceiling: np.ndarray  # (n_concepts,) Spearman–Brown corrected mean split-half r
    split_half_r: np.ndarray  # (n_concepts,) uncorrected mean split-half r
    split_r: np.ndarray  # (n_splits, n_concepts) r for every split
    n_splits: int
    method: str


def split_half_per_concept(
    ratings: np.ndarray, *, seed: int, n_splits: int = 200, method: str = "spearman"
) -> ConceptReliability:
    """Per-concept noise ceiling (see module docstring)."""
    if method not in ("spearman", "pearson"):
        raise ValueError("method must be 'spearman' or 'pearson'")
    R = _check(ratings)
    K, n, _ = R.shape
    rng = make_rng(seed)
    off = ~np.eye(n, dtype=bool)
    rs = np.full((n_splits, n), np.nan)
    for s in range(n_splits):
        h1, h2 = _halves(rng, K)
        with np.errstate(invalid="ignore"), _quiet():
            a = np.nanmean(R[h1], axis=0)[off].reshape(n, n - 1)
            b = np.nanmean(R[h2], axis=0)[off].reshape(n, n - 1)
        if not (np.isnan(a).any() or np.isnan(b).any()):
            rs[s] = _rowwise_corr(a, b, method)
        else:
            rs[s] = [_corr(a[i], b[i], method) for i in range(n)]
    with np.errstate(invalid="ignore"), _quiet():
        mean_r = np.nanmean(rs, axis=0)
    return ConceptReliability(np.asarray(spearman_brown(mean_r)), mean_r, rs, n_splits, method)


@dataclass
class DatasetCeiling:
    split_half: float  # Spearman–Brown corrected mean split-half r of the full matrix
    split_half_r: float  # uncorrected
    lower: float  # Nili lower bound (leave-one-subject-out)
    upper: float  # Nili upper bound (subject vs grand mean incl. itself)
    n_splits: int
    method: str


def dataset_noise_ceiling(
    ratings: np.ndarray, *, seed: int, n_splits: int = 200, method: str = "spearman"
) -> DatasetCeiling:
    """Dataset-level noise ceiling (see module docstring)."""
    R = _check(ratings)
    K, n, _ = R.shape
    iu = np.triu_indices(n, k=1)
    V = R[:, iu[0], iu[1]]  # (K, n_pairs)
    rng = make_rng(seed)
    rs = []
    with np.errstate(invalid="ignore"), _quiet():
        for _ in range(n_splits):
            h1, h2 = _halves(rng, K)
            rs.append(_corr(np.nanmean(V[h1], axis=0), np.nanmean(V[h2], axis=0), method))
        mean_r = float(np.nanmean(rs))
        grand = np.nanmean(V, axis=0)
        upper = np.nanmean([_corr(V[k], grand, method) for k in range(K)])
        lower = np.nanmean(
            [_corr(V[k], np.nanmean(np.delete(V, k, axis=0), axis=0), method) for k in range(K)]
        )
    return DatasetCeiling(float(spearman_brown(mean_r)), mean_r, float(lower), float(upper), n_splits, method)


class _quiet:
    """Silence 'Mean of empty slice' warnings from nanmean on all-NaN columns."""

    def __enter__(self):
        import warnings

        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


# Backwards-compatible alias used by the scaffold.
def split_half_ceiling(ratings: np.ndarray, *, seed: int, n_splits: int = 200) -> np.ndarray:
    return split_half_per_concept(ratings, seed=seed, n_splits=n_splits).ceiling
