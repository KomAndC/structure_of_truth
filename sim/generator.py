"""Synthetic subjects with known rotational / rescaling differences.

Shared space: ``X`` (N x k), N = 3 * n_per_layer concepts in layers 0 = physical,
1 = intersubjective, 2 = value-laden; drawn N(0, 1), centred and whitened so
``X^T X / N = I`` (no privileged axes, so GPA's principal axes carry no information about
the weighting axes).

Subject s:
* (A) rotation: the value-laden rows are rotated about the origin by
  ``R_s = expm(θ_s Ω_s)``, ``Ω_s`` a random skew-symmetric matrix with ||Ω||_F = sqrt(2)
  (so θ is a rotation angle in radians for a single-plane rotation),
  ``θ_s ~ U(0, theta_max)``;
* (B) rescaling: all coordinates are multiplied by ``exp(log_w_s / 2)`` per axis,
  ``log_w_sd = σ_s z_sd``, ``z`` standard normal centred within subject (geometric-mean
  weight 1), ``σ_s ~ U(0, sigma_max)``;
* (C) both (rotation first, then rescaling); "null": neither.

Dissimilarities: Euclidean distances of the subject's configuration plus symmetric
Gaussian noise with pair-specific s.d. ``noise * mean_dist(X) * (f_i + f_j) / 2`` where
``f`` is the layer noise factor (default 0.5 / 1.0 / 1.5), absolute value taken, zero
diagonal. ``noise`` is therefore a fraction of the mean true distance.

Ground truth magnitudes use the same definitions as the estimators in
``structure_of_truth.decompose``: ``θ_true = ||R_s - I||_F`` and ``b_true`` = row norms of
the double-centred log-weight matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.linalg import expm

from structure_of_truth.decompose import rescaling_magnitudes, rotation_magnitude

LAYERS = ("physical", "intersubjective", "value_laden")
GENERATORS = ("null", "A", "B", "C")


@dataclass(frozen=True)
class SimConfig:
    n_subjects: int = 50
    generator: str = "C"
    n_per_layer: int = 60
    k: int = 5
    noise: float = 0.3
    layer_noise: tuple[float, float, float] = (0.5, 1.0, 1.5)
    theta_max: float = 0.5
    sigma_max: float = 0.6

    def __post_init__(self):
        if self.generator not in GENERATORS:
            raise ValueError(f"generator must be one of {GENERATORS}")


@dataclass
class Population:
    """Everything needed to (re)measure the same subjects."""

    config: SimConfig
    X: np.ndarray  # (N, k)
    layer: np.ndarray  # (N,) layer index
    theta: np.ndarray  # (S,) rotation angles
    omega: np.ndarray  # (S, k, k) unit skew generators
    log_w: np.ndarray  # (S, k)
    mean_dist: float
    extra: dict = field(default_factory=dict)

    @property
    def value_idx(self) -> np.ndarray:
        return np.flatnonzero(self.layer == 2)

    @property
    def anchor_idx(self) -> np.ndarray:
        return np.flatnonzero(self.layer != 2)


def latent_space(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    X = rng.normal(size=(n, k))
    X -= X.mean(axis=0)
    L = np.linalg.cholesky(X.T @ X / n)
    return X @ np.linalg.inv(L).T


def random_skew(rng: np.random.Generator, k: int) -> np.ndarray:
    M = rng.normal(size=(k, k))
    O = M - M.T
    return O * (np.sqrt(2.0) / np.linalg.norm(O))


def make_population(cfg: SimConfig, rng: np.random.Generator) -> Population:
    N, S, k = 3 * cfg.n_per_layer, cfg.n_subjects, cfg.k
    X = latent_space(rng, N, k)
    layer = np.repeat(np.arange(3), cfg.n_per_layer)
    omega = np.stack([random_skew(rng, k) for _ in range(S)])
    theta = rng.uniform(0.0, cfg.theta_max, size=S)
    z = rng.normal(size=(S, k))
    z -= z.mean(axis=1, keepdims=True)
    log_w = rng.uniform(0.0, cfg.sigma_max, size=S)[:, None] * z
    if cfg.generator in ("null", "B"):
        theta = np.zeros(S)
    if cfg.generator in ("null", "A"):
        log_w = np.zeros((S, k))
    D0 = np.linalg.norm(X[:, None] - X[None], axis=-1)
    mean_dist = float(D0[np.triu_indices(N, 1)].mean())
    return Population(cfg, X, layer, theta, omega, log_w, mean_dist)


def subject_config(pop: Population, s: int, theta: float | None = None, log_w: np.ndarray | None = None) -> np.ndarray:
    theta = pop.theta[s] if theta is None else theta
    log_w = pop.log_w[s] if log_w is None else log_w
    Y = pop.X.copy()
    v = pop.value_idx
    Y[v] = pop.X[v] @ expm(theta * pop.omega[s])
    return Y * np.exp(log_w / 2.0)


def true_magnitudes(pop: Population, theta: np.ndarray | None = None, log_w: np.ndarray | None = None):
    theta = pop.theta if theta is None else theta
    log_w = pop.log_w if log_w is None else log_w
    th = np.array([rotation_magnitude(expm(t * O)) for t, O in zip(theta, pop.omega)])
    return th, rescaling_magnitudes(log_w)


def measure(
    pop: Population,
    rng: np.random.Generator,
    theta: np.ndarray | None = None,
    log_w: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One noisy dissimilarity matrix per subject. Returns (D_noisy, D_true), (S, N, N)."""
    cfg = pop.config
    theta = pop.theta if theta is None else theta
    log_w = pop.log_w if log_w is None else log_w
    f = np.asarray(cfg.layer_noise)[pop.layer]
    sd = cfg.noise * pop.mean_dist * (f[:, None] + f[None, :]) / 2.0
    N = len(pop.layer)
    S = len(theta)
    D_true = np.empty((S, N, N))
    D = np.empty((S, N, N))
    for s in range(S):
        Y = subject_config(pop, s, theta[s], log_w[s])
        D_true[s] = np.linalg.norm(Y[:, None] - Y[None], axis=-1)
        E = rng.normal(size=(N, N))
        E = np.triu(E, 1)
        E = E + E.T
        D[s] = np.abs(D_true[s] + sd * E)
        np.fill_diagonal(D[s], 0.0)
    return D, D_true


__all__ = [
    "GENERATORS",
    "LAYERS",
    "Population",
    "SimConfig",
    "make_population",
    "measure",
    "replace",
    "subject_config",
    "true_magnitudes",
]
