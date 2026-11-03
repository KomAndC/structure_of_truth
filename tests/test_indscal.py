import itertools

import numpy as np
import pytest

from conftest import indscal_data, pairwise, random_orthogonal
from structure_of_truth.indscal import indscal, subject_distances


def _match(X_true, X_hat):
    """Best permutation/sign match of columns; returns (congruences, perm)."""
    dim = X_true.shape[1]
    Xt = X_true - X_true.mean(0)
    best = None
    for perm in itertools.permutations(range(dim)):
        cong = np.array([
            abs(Xt[:, d] @ X_hat[:, p]) / (np.linalg.norm(Xt[:, d]) * np.linalg.norm(X_hat[:, p]))
            for d, p in enumerate(perm)
        ])
        if best is None or cong.min() > best[0].min():
            best = (cong, perm)
    return best


@pytest.mark.parametrize("dim", [2, 3])
def test_noise_free_recovery(dim):
    X, W, Ds = indscal_data(n=12, dim=dim, K=6, seed=dim)
    res = indscal(Ds, dim=dim)
    assert res.fit > 1 - 1e-8
    cong, perm = _match(X, res.X)
    assert cong.min() > 0.9999
    # weights recovered up to the per-dimension scale fixed by the normalisation
    Xt = X - X.mean(0)
    for d, p in enumerate(perm):
        scale = (Xt[:, d] ** 2).sum() / X.shape[0]
        assert np.allclose(res.weights[:, p], W[:, d] * scale, rtol=1e-4)
    # model distances reproduce the data
    for D, Dhat in zip(Ds, subject_distances(res.X, res.weights)):
        assert np.allclose(D, Dhat, atol=1e-5)


def test_normalisation_is_as_documented():
    _, _, Ds = indscal_data(noise=0.1, seed=4)
    res = indscal(Ds, dim=2)
    n = res.X.shape[0]
    assert np.allclose(res.X.mean(axis=0), 0, atol=1e-10)
    assert np.allclose((res.X**2).sum(axis=0) / n, 1.0)
    assert np.all(res.weights >= 0)
    assert np.all(np.diff(res.weights.mean(axis=0)) <= 0)  # sorted by mean weight
    assert res.converged


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("init", ["svd", "random"])
def test_als_strain_never_increases(seed, init):
    _, _, Ds = indscal_data(n=15, dim=3, K=8, seed=seed, noise=0.5)
    res = indscal(Ds, dim=3, init=init, seed=seed, max_iter=500)
    h = np.array(res.loss_history)
    assert len(h) > 2
    assert np.all(np.diff(h) <= 1e-10 * h[0]), np.diff(h).max()


def test_invariance_to_rotation_of_subject_configs():
    # Distances are what INDSCAL sees: rigidly moving each subject's private space
    # must not change the solution.
    X, W, Ds = indscal_data(seed=7, noise=0.0)
    Ds_rot = [pairwise((X * np.sqrt(w)) @ random_orthogonal(2, 50 + k) + 3.0) for k, w in enumerate(W)]
    a, b = indscal(Ds, dim=2), indscal(Ds_rot, dim=2)
    assert np.allclose(a.X, b.X, atol=1e-6)
    assert np.allclose(a.weights, b.weights, rtol=1e-5)


def test_scale_of_input_goes_into_weights():
    _, _, Ds = indscal_data(seed=8, noise=0.2)
    a = indscal(Ds, dim=2)
    b = indscal([3.0 * D for D in Ds], dim=2)
    assert np.allclose(a.X, b.X, atol=1e-6)
    assert np.allclose(b.weights, 9.0 * a.weights, rtol=1e-5)
    assert b.fit == pytest.approx(a.fit, abs=1e-9)


def test_random_init_requires_seed_and_is_reproducible():
    _, _, Ds = indscal_data(seed=9, noise=0.3)
    with pytest.raises(ValueError):
        indscal(Ds, dim=2, init="random")
    a = indscal(Ds, dim=2, init="random", seed=1)
    b = indscal(Ds, dim=2, init="random", seed=1)
    assert np.array_equal(a.X, b.X)
