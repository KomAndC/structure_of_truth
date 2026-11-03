"""Generalized Procrustes analysis (GPA).

Model: each subject k supplies a configuration ``X_k`` (n_concepts x dim). GPA finds a
translation ``t_k``, an orthogonal map ``R_k`` and (optionally) an isotropic scale ``s_k``
per subject so that the transformed configurations ``Y_k = s_k (X_k - 1 t_k) R_k`` are as
close as possible to their mean ``G`` (the consensus / shared space)::

    minimise  L = sum_k || Y_k - G ||_F^2 ,   G = mean_k Y_k

Normalisation
-------------
* Translation: every configuration is column-centred (concept centroid at the origin).
* ``scaling=True``: each centred configuration is first scaled to unit Frobenius norm, and
  the scales are then re-estimated subject to ``sum_k s_k^2 ||X_k||^2 = K`` (Ten Berge
  1977): with unit-norm inputs the optimal ``s`` is ``sqrt(K)`` times the leading
  eigenvector of the inner-product matrix ``C_kl = <Y_k, Y_l>``. This constraint excludes
  the trivial solution ``s = 0``.
* ``scaling=False``: configurations keep their original units (only centred and rotated);
  the residual is then in the units of the input.

Algorithm and convergence
-------------------------
Initial target is subject 0 (deterministic); the first sweep rotates onto it. Each later
iteration (1) rotates every ``Y_k`` onto
the current consensus by orthogonal Procrustes (SVD), (2) updates scales if requested,
(3) recomputes ``G``. Each step minimises ``L`` exactly over its block of parameters, so
``L`` is non-increasing. Iteration stops when the relative decrease
``(L_prev - L) / L_0`` falls below ``tol`` or ``max_iter`` is reached.

Configurations with different numbers of columns are zero-padded to the widest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class GPAResult:
    consensus: np.ndarray  # (n_concepts, dim) G = mean of aligned configurations
    aligned: np.ndarray  # (n_subjects, n_concepts, dim) Y_k
    residuals: np.ndarray  # (n_subjects, n_concepts, dim) Y_k - G
    rotations: np.ndarray  # (n_subjects, dim, dim) R_k
    scales: np.ndarray  # (n_subjects,) s_k applied to the *centred original* X_k
    translations: np.ndarray  # (n_subjects, dim) t_k (column means of X_k)
    residual_ss_per_subject: np.ndarray  # (n_subjects,)
    residual_ss_per_concept: np.ndarray  # (n_concepts,)
    residual_ss: float  # total L
    n_iter: int
    converged: bool
    loss_history: list[float] = field(default_factory=list)


def orthogonal_procrustes(
    X: np.ndarray, target: np.ndarray, *, allow_reflection: bool = True
) -> np.ndarray:
    """Orthogonal R minimising ||X R - target||_F.

    With ``allow_reflection=False`` the solution is restricted to det(R) = +1.
    """
    U, _, Vt = np.linalg.svd(X.T @ target)
    if not allow_reflection and np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    return U @ Vt


def _pad(configs: Sequence[np.ndarray]) -> np.ndarray:
    arrs = [np.asarray(c, dtype=float) for c in configs]
    if len(arrs) < 2:
        raise ValueError("GPA needs at least 2 configurations")
    n = arrs[0].shape[0]
    if any(a.ndim != 2 or a.shape[0] != n for a in arrs):
        raise ValueError("all configurations must be 2-D with the same number of concepts")
    dim = max(a.shape[1] for a in arrs)
    out = np.zeros((len(arrs), n, dim))
    for k, a in enumerate(arrs):
        out[k, :, : a.shape[1]] = a
    return out


def _loss(Y: np.ndarray) -> float:
    return float(((Y - Y.mean(axis=0)) ** 2).sum())


def generalized_procrustes(
    configs: Sequence[np.ndarray],
    *,
    scaling: bool = True,
    allow_reflection: bool = True,
    tol: float = 1e-12,
    max_iter: int = 1000,
) -> GPAResult:
    """Align a list of subject configurations to a common consensus. See module docstring."""
    X = _pad(configs)
    K = X.shape[0]
    translations = X.mean(axis=1)
    Xc = X - translations[:, None, :]
    norms = np.linalg.norm(Xc, axis=(1, 2))
    if np.any(norms == 0):
        raise ValueError("degenerate configuration (all points identical)")

    if scaling:
        Z = Xc / norms[:, None, None]  # unit-norm working copies
        s = np.ones(K)
    else:
        Z = Xc
        s = np.ones(K)

    dim = X.shape[2]
    R = np.repeat(np.eye(dim)[None], K, axis=0)
    Y = s[:, None, None] * Z
    target = Y[0]
    history: list[float] = []  # loss after each sweep (monotone from sweep 1 on)
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        R = np.stack([orthogonal_procrustes(Z[k], target, allow_reflection=allow_reflection) for k in range(K)])
        ZR = np.einsum("snd,sde->sne", Z, R)
        if scaling:
            C = np.einsum("snd,tnd->st", ZR, ZR)
            evals, evecs = np.linalg.eigh(C)
            phi = evecs[:, -1]
            phi = phi if phi.sum() >= 0 else -phi
            s = np.sqrt(K) * phi
        Y = s[:, None, None] * ZR
        target = Y.mean(axis=0)
        history.append(_loss(Y))
        if len(history) >= 2 and (history[-2] - history[-1]) <= tol * max(history[0], np.finfo(float).tiny):
            converged = True
            break

    G = Y.mean(axis=0)
    resid = Y - G
    scales = s / norms if scaling else np.ones(K)
    return GPAResult(
        consensus=G,
        aligned=Y,
        residuals=resid,
        rotations=R,
        scales=scales,
        translations=translations,
        residual_ss_per_subject=(resid**2).sum(axis=(1, 2)),
        residual_ss_per_concept=(resid**2).sum(axis=(0, 2)),
        residual_ss=float((resid**2).sum()),
        n_iter=n_iter,
        converged=converged,
        loss_history=history,
    )
