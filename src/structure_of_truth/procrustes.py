"""Generalized Procrustes analysis (rotation-only alignment to a shared space)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GPAResult:
    consensus: np.ndarray  # (n_concepts, dim) shared space
    aligned: np.ndarray  # (n_subjects, n_concepts, dim) configurations after alignment
    rotations: np.ndarray  # (n_subjects, dim, dim) per-subject orthogonal maps
    residual_ss: float  # total sum of squares of aligned - consensus
    n_iter: int


def _center_scale(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(X)
    if norm == 0:
        raise ValueError("degenerate configuration (all points identical)")
    return X / norm


def orthogonal_procrustes(X: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Orthogonal R minimising ||X R - target||_F (reflections allowed)."""
    U, _, Vt = np.linalg.svd(X.T @ target)
    return U @ Vt


def generalized_procrustes(
    configs: np.ndarray, *, tol: float = 1e-10, max_iter: int = 500
) -> GPAResult:
    """Align subject configurations (n_subjects, n_concepts, dim) to a common consensus.

    Each configuration is centred and scaled to unit Frobenius norm, then iteratively
    rotated onto the running mean. Deterministic: initial target is subject 0.
    """
    configs = np.asarray(configs, dtype=float)
    if configs.ndim != 3:
        raise ValueError("configs must have shape (n_subjects, n_concepts, dim)")
    Xs = np.stack([_center_scale(X) for X in configs])
    target = Xs[0].copy()
    prev = np.inf
    rotations = np.repeat(np.eye(Xs.shape[2])[None], Xs.shape[0], axis=0)
    for it in range(1, max_iter + 1):
        rotations = np.stack([orthogonal_procrustes(X, target) for X in Xs])
        aligned = np.einsum("snd,sde->sne", Xs, rotations)
        target = _center_scale(aligned.mean(axis=0))
        ss = float(((aligned - aligned.mean(axis=0)) ** 2).sum())
        if abs(prev - ss) < tol:
            break
        prev = ss
    consensus = aligned.mean(axis=0)
    residual_ss = float(((aligned - consensus) ** 2).sum())
    return GPAResult(consensus, aligned, rotations, residual_ss, it)
