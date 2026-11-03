"""INDSCAL (individual differences scaling) by alternating least squares.

Model (Carroll & Chang 1970)
----------------------------
Each subject k supplies a symmetric dissimilarity matrix ``D_k`` (n x n). It is turned into
a scalar-product matrix by classical double centring::

    B_k = -1/2 J (D_k ∘ D_k) J ,   J = I - 11^T / n

and modelled as ``B_k ≈ X W_k X^T`` with a shared space ``X`` (n x dim) and non-negative
diagonal per-axis weights ``W_k = diag(w_k)``. Subject k's private space is
``X W_k^{1/2}``; its model distances are ``d_ij^(k) = ||(x_i - x_j) ∘ sqrt(w_k)||``.

Loss
----
The ALS objective is the INDSCAL *strain* (least-squares loss on scalar products)::

    strain(A, B, W) = sum_k || B_k - A W_k B^T ||_F^2

fitted CANDECOMP-style with two concept-mode factors ``A`` and ``B`` (the standard
Carroll–Chang algorithm). Each update is an exact (constrained) least-squares minimiser
for its block — ``A`` and ``B`` by ordinary LS, the weights by NNLS (``w_k >= 0``) — so the
strain is non-increasing over iterations (up to floating-point round-off). After
convergence the symmetric solution ``X`` is taken as ``(A + B) / 2``; ``A`` and ``B``
coincide at a symmetric optimum, and the remaining asymmetry ``max|A - B|`` is reported.
The final ``strain`` and ``loss_history`` refer to the ALS objective; ``symmetric_strain``
is the loss of the symmetrised solution ``X W_k X^T``. Note this is *not* Kruskal stress on
distances; use :mod:`structure_of_truth.stress` with :func:`subject_distances` for that.

Normalisation (identification)
------------------------------
The model is invariant to per-dimension rescaling ``X -> X S``, ``W_k -> S^{-2} W_k``. We
fix the scale of the shared space and put all scale into the weights:

* every column of ``X`` (and of ``A``, ``B`` during fitting) is centred (automatic,
  because ``B_k`` is double centred) and scaled to sum of squares ``n``, i.e.
  ``diag(X^T X) / n = 1`` (unit variance per dimension);
* the weights absorb the removed scale: ``w_kd <- w_kd * ||a_d|| ||b_d|| / n``;
* dimensions are sorted by decreasing mean weight and each column's sign is chosen so
  its largest-magnitude entry is positive.

Rescaling does not change the strain, so it is applied every iteration without breaking
monotonicity. Unlike GPA, INDSCAL dimensions are identified up to permutation and sign
only (no rotational indeterminacy) when weights differ across subjects.

Convergence criterion
---------------------
Stop when the relative strain decrease ``(strain_prev - strain) / strain_0`` is below
``tol`` (default 1e-10), or after ``max_iter`` iterations. ``strain_0`` is the strain after
the initial weight fit.

Initialisation
--------------
``init="svd"`` (default, deterministic): classical MDS of the mean ``B_k``, i.e. leading
eigenvectors scaled by sqrt(eigenvalue). ``init="random"``: Gaussian factors drawn from
``make_rng(seed)``; ``seed`` is then required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import nnls

from .seeds import make_rng


@dataclass
class INDSCALResult:
    X: np.ndarray  # (n, dim) shared space, columns centred, diag(X^T X)/n = 1
    weights: np.ndarray  # (n_subjects, dim) non-negative per-axis weights w_kd
    strain: float  # final ALS objective
    symmetric_strain: float  # sum_k ||B_k - X W_k X^T||^2
    fit: float  # 1 - strain / sum_k ||B_k||^2  (variance-accounted-for in scalar products)
    asymmetry: float  # max |A - B| at convergence
    n_iter: int
    converged: bool
    loss_history: list[float] = field(default_factory=list)


def double_center(D: np.ndarray) -> np.ndarray:
    """Classical-MDS scalar products: -1/2 J (D∘D) J."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * J @ (D**2) @ J


def _stack(dissimilarities: Sequence[np.ndarray]) -> np.ndarray:
    Ds = [np.asarray(d, dtype=float) for d in dissimilarities]
    if len(Ds) < 2:
        raise ValueError("INDSCAL needs at least 2 subjects")
    n = Ds[0].shape[0]
    for d in Ds:
        if d.shape != (n, n):
            raise ValueError("all dissimilarity matrices must be n x n with the same n")
        if np.isnan(d).any():
            raise ValueError("missing values are not supported (see SPEC open decisions)")
        if not np.allclose(d, d.T):
            raise ValueError("dissimilarity matrices must be symmetric")
    return np.stack([double_center(d) for d in Ds], axis=-1)  # (n, n, K)


def _strain(T: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    R = T - np.einsum("id,jd,kd->ijk", A, B, C)
    return float((R**2).sum())


def _nnls_weights(T: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Exact NNLS for each subject's weights: min_c>=0 ||vec(B_k) - (A⊙B) c||^2.

    Solved in the dim-dimensional normal-equation space: with G = (A^T A)∘(B^T B) =
    V S V^T, the problem equals min ||S^{1/2} V^T c - S^{-1/2} V^T r_k||^2 + const where
    r_k = (A⊙B)^T vec(B_k).
    """
    G = (A.T @ A) * (B.T @ B)
    s, V = np.linalg.eigh(G)
    keep = s > s.max() * 1e-12
    s, V = s[keep], V[:, keep]
    M = np.sqrt(s)[:, None] * V.T
    r = np.einsum("ijk,id,jd->kd", T, A, B)
    rhs = (r @ V) / np.sqrt(s)
    return np.stack([nnls(M, rhs[k])[0] for k in range(T.shape[2])])


def _ls_factor(T: np.ndarray, other: np.ndarray, C: np.ndarray, mode: int) -> np.ndarray:
    G = (other.T @ other) * (C.T @ C)
    if mode == 0:
        rhs = np.einsum("ijk,jd,kd->id", T, other, C)
    else:
        rhs = np.einsum("ijk,id,kd->jd", T, other, C)
    return rhs @ np.linalg.pinv(G)


def _normalise(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, ...]:
    n = A.shape[0]
    na, nb = np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0)
    na[na == 0] = 1.0
    nb[nb == 0] = 1.0
    A = A / na * np.sqrt(n)
    B = B / nb * np.sqrt(n)
    C = C * (na * nb / n)
    return A, B, C


def indscal(
    dissimilarities: Sequence[np.ndarray],
    dim: int,
    *,
    init: str = "svd",
    seed: int | None = None,
    tol: float = 1e-10,
    max_iter: int = 2000,
) -> INDSCALResult:
    """Fit INDSCAL by ALS. See the module docstring for loss, normalisation and stopping."""
    T = _stack(dissimilarities)
    n, _, K = T.shape
    if not 1 <= dim < n:
        raise ValueError("dim must satisfy 1 <= dim < n_concepts")

    if init == "svd":
        evals, evecs = np.linalg.eigh(T.mean(axis=2))
        idx = np.argsort(evals)[::-1][:dim]
        A = evecs[:, idx] * np.sqrt(np.clip(evals[idx], 1e-12, None))
    elif init == "random":
        if seed is None:
            raise ValueError("init='random' requires an explicit seed")
        A = make_rng(seed).normal(size=(n, dim))
        A -= A.mean(axis=0)
    else:
        raise ValueError("init must be 'svd' or 'random'")
    B = A.copy()
    A, B, _ = _normalise(A, B, np.ones((K, dim)))
    C = _nnls_weights(T, A, B)

    history = [_strain(T, A, B, C)]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        A = _ls_factor(T, B, C, mode=0)
        B = _ls_factor(T, A, C, mode=1)
        C = _nnls_weights(T, A, B)
        A, B, C = _normalise(A, B, C)
        history.append(_strain(T, A, B, C))
        if (history[-2] - history[-1]) <= tol * max(history[0], np.finfo(float).tiny):
            converged = True
            break

    asym = float(np.abs(A - B).max())
    X = (A + B) / 2.0
    X = X / np.linalg.norm(X, axis=0) * np.sqrt(n)
    W = _nnls_weights(T, X, X)  # weights refitted for the symmetric solution

    order = np.argsort(-W.mean(axis=0))
    X, W = X[:, order], W[:, order]
    signs = np.sign(X[np.abs(X).argmax(axis=0), np.arange(dim)])
    signs[signs == 0] = 1.0
    X = X * signs

    total = float((T**2).sum())
    return INDSCALResult(
        X=X,
        weights=W,
        strain=history[-1],
        symmetric_strain=_strain(T, X, X, W),
        fit=1.0 - history[-1] / total if total > 0 else 1.0,
        asymmetry=asym,
        n_iter=it,
        converged=converged,
        loss_history=history,
    )


def subject_distances(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Model distances for every subject: (n_subjects, n, n), d = ||(x_i-x_j)∘sqrt(w_k)||."""
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.einsum("ijd,kd->kij", diff**2, weights))
