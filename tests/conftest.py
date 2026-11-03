import numpy as np
from scipy.stats import special_ortho_group, ortho_group

from structure_of_truth.seeds import make_rng


def random_orthogonal(dim: int, seed: int, reflect: bool = False) -> np.ndarray:
    if reflect:
        return ortho_group.rvs(dim, random_state=seed)
    return special_ortho_group.rvs(dim, random_state=seed)


def pairwise(X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X[:, None] - X[None], axis=-1)


def indscal_data(n=12, dim=2, K=6, seed=0, noise=0.0):
    """Distances from X diag(sqrt(w_k)) with distinct weights; optional symmetric noise."""
    rng = make_rng(seed)
    X = rng.normal(size=(n, dim))
    W = rng.uniform(0.2, 2.0, size=(K, dim))
    Ds = []
    for k in range(K):
        D = pairwise(X * np.sqrt(W[k]))
        if noise:
            E = noise * rng.normal(size=(n, n))
            D = np.abs(D + (E + E.T) / 2)
            np.fill_diagonal(D, 0)
        Ds.append(D)
    return X, W, Ds

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
