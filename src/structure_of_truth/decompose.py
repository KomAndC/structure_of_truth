"""Split between-subject differences into a rotational and a rescaling component.

Target model (SPEC, prediction 3): subject s sees the shared space ``X`` with
  (A) the *value-laden block* of concepts rotated by an orthogonal ``R_s`` relative to the
      rest ("definitional" difference), and/or
  (B) every axis rescaled by per-axis weights ``w_s`` ("value weighting"):
      ``X_s = X_rot(s) diag(sqrt(w_s))``.

Estimators (per subject), both starting from a classical-MDS configuration ``Y_s`` of the
subject's dissimilarity matrix in ``dim`` dimensions:

INDSCAL route (``*_indscal``)
    Fit INDSCAL to all subjects -> shared ``X̂``, weights ``ŵ_s``.
    *Rescaling*: ``b_s = || C(log ŵ)_s ||`` where ``C`` double-centres the
    subjects x axes matrix of log weights (removes each subject's overall size and each
    axis' common weight, which INDSCAL's normalisation cannot separate from ``X``).
    *Rotation*: align ``Y_s`` to the subject's INDSCAL space ``X̂ diag(sqrt ŵ_s)`` by a
    similarity transform fitted on the *anchor* concepts only, then fit a proper rotation
    of the centred block of ``X̂`` *before* the subject's axis scaling,
    ``min_R ||X̂_b R diag(sqrt ŵ_s) - Y_b||`` (:func:`weighted_block_rotation`):
    ``θ_s = ||R̂_s - I||_F`` (chordal rotation magnitude). (Dividing the subject by
    ``sqrt ŵ_s`` instead was tried first and amplified noise on down-weighted axes, which
    leaked rescaling into the rotation estimate.)

GPA route (``*_gpa``), for comparison (no per-axis model)
    GPA (with scaling) of all ``Y_s`` -> consensus ``G``. Rotation as above but against
    ``G`` and without un-weighting. Rescaling: per-axis least-squares scale of the
    anchor-aligned subject along the principal axes of ``G``; then the same ``b_s``.
    The principal axes of ``G`` need not be the axes the weights act on, so this is
    expected to be the weaker estimator.

Assumptions (see SPEC open decisions): the dimensionality is known; the block suspected of
rotating and a set of anchor concepts assumed *not* to rotate are specified in advance.
Both magnitudes have a positive noise floor (they are norms), so they must be compared
against a null calibrated to the measurement noise, never against zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .indscal import INDSCALResult, double_center, indscal
from .procrustes import GPAResult, generalized_procrustes, orthogonal_procrustes


def classical_mds(D: np.ndarray, dim: int) -> np.ndarray:
    """Torgerson classical MDS: top-``dim`` eigenvectors of the double-centred matrix."""
    evals, evecs = np.linalg.eigh(double_center(D))
    idx = np.argsort(evals)[::-1][:dim]
    return evecs[:, idx] * np.sqrt(np.clip(evals[idx], 0.0, None))


def similarity_align(Y: np.ndarray, ref: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Translate/rotate(/reflect)/scale all of ``Y`` so that ``Y[rows]`` best matches ``ref[rows]``."""
    mu_y, mu_r = Y[rows].mean(axis=0), ref[rows].mean(axis=0)
    Yc, Rc = Y[rows] - mu_y, ref[rows] - mu_r
    R = orthogonal_procrustes(Yc, Rc)
    s = np.trace((Yc @ R).T @ Rc) / max((Yc**2).sum(), np.finfo(float).tiny)
    return s * (Y - mu_y) @ R + mu_r


def block_rotation(ref: np.ndarray, Y: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Proper rotation R minimising ||ref_c[block] R - Y_c[block]|| (both block-centred)."""
    a = ref[block] - ref[block].mean(axis=0)
    b = Y[block] - Y[block].mean(axis=0)
    return orthogonal_procrustes(a, b, allow_reflection=False)


def weighted_block_rotation(
    ref: np.ndarray, Y: np.ndarray, block: np.ndarray, d: np.ndarray, *, n_iter: int = 200, tol: float = 1e-10
) -> np.ndarray:
    """Proper rotation R minimising ||A R diag(d) - B||_F with A, B the centred blocks.

    The rotation acts in the shared space *before* the subject's axis scaling ``d``
    (= sqrt(weights)), so noise on down-weighted axes is not amplified by dividing by
    ``d``. No closed form exists; we use the majorization update of Kiers (1990):
    ``R <- polar(A^T B D + a δ R - A^T A R D^2)`` with ``a = λmax(A^T A)``, ``δ = max d^2``,
    which never increases the loss. Started from the unweighted solution.
    """
    A = ref[block] - ref[block].mean(axis=0)
    B = Y[block] - Y[block].mean(axis=0)
    D = np.diag(d)
    AtA, AtBD = A.T @ A, A.T @ B @ D
    a, delta = np.linalg.eigvalsh(AtA)[-1], float((d**2).max())
    R = orthogonal_procrustes(A, B / d, allow_reflection=False)
    prev = np.inf
    for _ in range(n_iter):
        M = AtBD + a * delta * R - AtA @ R @ D @ D
        U, _, Vt = np.linalg.svd(M)
        if np.linalg.det(U @ Vt) < 0:
            U[:, -1] *= -1
        R = U @ Vt
        loss = float(((A @ R @ D - B) ** 2).sum())
        if prev - loss <= tol * max(loss, np.finfo(float).tiny):
            break
        prev = loss
    return R


def rotation_magnitude(R: np.ndarray) -> float:
    """Chordal distance from the identity, ||R - I||_F."""
    return float(np.linalg.norm(R - np.eye(R.shape[0])))


def rescaling_magnitudes(log_w: np.ndarray) -> np.ndarray:
    """Row norms of the double-centred (subjects x axes) log-weight matrix."""
    L = log_w - log_w.mean(axis=1, keepdims=True)
    L = L - L.mean(axis=0, keepdims=True)
    return np.linalg.norm(L, axis=1)


@dataclass
class Components:
    theta_indscal: np.ndarray  # (S,)
    b_indscal: np.ndarray  # (S,)
    theta_gpa: np.ndarray  # (S,)
    b_gpa: np.ndarray  # (S,)
    indscal: INDSCALResult
    gpa: GPAResult


def estimate_components(
    dissimilarities: Sequence[np.ndarray],
    dim: int,
    *,
    block: np.ndarray,
    anchors: np.ndarray,
    weight_floor: float = 1e-3,
    indscal_kwargs: dict | None = None,
) -> Components:
    """Per-subject rotational and rescaling magnitudes by the INDSCAL and GPA routes."""
    Ds = [np.asarray(d, dtype=float) for d in dissimilarities]
    block, anchors = np.asarray(block), np.asarray(anchors)
    Ys = [classical_mds(D, dim) for D in Ds]

    ind = indscal(Ds, dim, **(indscal_kwargs or {}))
    W = np.maximum(ind.weights, weight_floor * max(ind.weights.mean(), np.finfo(float).tiny))
    theta_i = np.empty(len(Ds))
    for s, Y in enumerate(Ys):
        d = np.sqrt(W[s])
        Z = similarity_align(Y, ind.X * d, anchors)
        theta_i[s] = rotation_magnitude(weighted_block_rotation(ind.X, Z, block, d))
    b_i = rescaling_magnitudes(np.log(W))

    gpa = generalized_procrustes(Ys, scaling=True)
    G = gpa.consensus
    _, _, Vt = np.linalg.svd(G - G.mean(axis=0), full_matrices=False)
    g = (G - G.mean(axis=0)) @ Vt.T
    theta_g = np.empty(len(Ds))
    log_s = np.empty((len(Ds), dim))
    for s, Y in enumerate(Ys):
        Ya = similarity_align(Y, G, anchors)
        theta_g[s] = rotation_magnitude(block_rotation(G, Ya, block))
        p = (Ya - Ya.mean(axis=0)) @ Vt.T
        scale = (p * g).sum(axis=0) / np.maximum((g**2).sum(axis=0), np.finfo(float).tiny)
        log_s[s] = np.log(np.maximum(np.abs(scale), weight_floor))
    b_g = rescaling_magnitudes(2.0 * log_s)  # weights are squared axis scales

    return Components(theta_i, b_i, theta_g, b_g, ind, gpa)
