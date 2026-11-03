import numpy as np
import pytest
from scipy.spatial import procrustes as scipy_procrustes

from conftest import pairwise, random_orthogonal
from structure_of_truth.procrustes import generalized_procrustes
from structure_of_truth.seeds import make_rng


def _base(seed=0, n=20, dim=3):
    return make_rng(seed).normal(size=(n, dim))


def _transformed(base, K, seed, scale=True, reflect=True):
    rng = make_rng(seed)
    out = []
    for k in range(K):
        R = random_orthogonal(base.shape[1], seed + k, reflect=reflect and k % 2 == 1)
        s = rng.uniform(0.3, 3.0) if scale else 1.0
        t = rng.normal(size=base.shape[1]) * 5
        out.append(s * base @ R + t)
    return out


def test_noise_free_recovery_with_scaling():
    base = _base()
    res = generalized_procrustes(_transformed(base, 5, seed=10))
    assert res.converged
    assert res.residual_ss < 1e-20
    # consensus is the base shape up to a similarity transform
    D_true, D_cons = pairwise(base), pairwise(res.consensus)
    assert np.allclose(D_cons / D_cons.max(), D_true / D_true.max(), atol=1e-10)


def test_noise_free_recovery_without_scaling_same_scale():
    base = _base()
    res = generalized_procrustes(_transformed(base, 4, seed=3, scale=False), scaling=False)
    assert res.residual_ss < 1e-20
    assert np.allclose(pairwise(res.consensus), pairwise(base), atol=1e-10)


def test_without_scaling_size_differences_remain_in_residual():
    base = _base()
    res = generalized_procrustes([base, 2 * base], scaling=False)
    assert res.residual_ss > 1.0


def test_no_reflection_option():
    base = _base()
    mirrored = base * np.array([1, 1, -1])
    assert generalized_procrustes([base, mirrored]).residual_ss < 1e-20
    assert generalized_procrustes([base, mirrored], allow_reflection=False).residual_ss > 1e-3
    for R in generalized_procrustes([base, mirrored], allow_reflection=False).rotations:
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_invariance_to_rotation_translation_scale_of_inputs():
    rng = make_rng(1)
    base = _base()
    noisy = [base + 0.2 * rng.normal(size=base.shape) for _ in range(6)]
    ref = generalized_procrustes(noisy)
    moved = []
    for k, X in enumerate(noisy):
        moved.append(rng.uniform(0.2, 5.0) * X @ random_orthogonal(3, 100 + k) + rng.normal(size=3))
    res = generalized_procrustes(moved)
    assert res.residual_ss == pytest.approx(ref.residual_ss, rel=1e-8)
    assert np.allclose(res.residual_ss_per_concept, ref.residual_ss_per_concept, rtol=1e-6, atol=1e-12)
    assert np.allclose(pairwise(res.consensus), pairwise(ref.consensus), atol=1e-8)


def test_loss_history_non_increasing_and_residual_bookkeeping():
    rng = make_rng(2)
    base = _base()
    configs = [base + 0.5 * rng.normal(size=base.shape) for _ in range(7)]
    for scaling in (True, False):
        res = generalized_procrustes(configs, scaling=scaling)
        h = np.array(res.loss_history)
        assert np.all(np.diff(h) <= 1e-12 * h[0])
        assert res.residuals.shape == (7, 20, 3)
        assert np.allclose(res.residuals.mean(axis=0), 0, atol=1e-12)
        assert res.residual_ss_per_subject.sum() == pytest.approx(res.residual_ss)
        assert res.residual_ss_per_concept.sum() == pytest.approx(res.residual_ss)


def test_aligned_equals_transformed_input():
    rng = make_rng(4)
    configs = [rng.normal(size=(10, 2)) for _ in range(3)]
    res = generalized_procrustes(configs)
    for k, X in enumerate(configs):
        Y = res.scales[k] * (X - res.translations[k]) @ res.rotations[k]
        assert np.allclose(Y, res.aligned[k])


def test_different_dimensionalities_are_zero_padded():
    base = _base(dim=2)
    res = generalized_procrustes([base, np.column_stack([base, np.zeros(20)])])
    assert res.consensus.shape == (20, 3)
    assert res.residual_ss < 1e-20


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_two_subject_case_matches_scipy(seed):
    rng = make_rng(seed)
    A = rng.normal(size=(15, 3))
    B = 2.5 * (A + 0.3 * rng.normal(size=A.shape)) @ random_orthogonal(3, seed) + 4.0
    mtx1, mtx2, disparity = scipy_procrustes(A, B)

    res = generalized_procrustes([A, B])
    # symmetric GPA residual L = 1 - c, scipy disparity = 1 - c^2 (c = Procrustes cosine)
    c = 1.0 - res.residual_ss
    assert disparity == pytest.approx(1.0 - c**2, rel=1e-9)
    # same relative alignment: express subject 1 in subject 0's (scipy's) frame
    Rrel = res.rotations[1] @ res.rotations[0].T
    B_in_A = (B - B.mean(0)) / np.linalg.norm(B - B.mean(0)) @ Rrel
    assert np.allclose(B_in_A * c, mtx2, atol=1e-10)
    assert np.allclose(res.aligned[0] @ res.rotations[0].T, mtx1, atol=1e-10)

    # unscaled GPA finds the same rotation
    res_ns = generalized_procrustes([A, B], scaling=False)
    Rrel_ns = res_ns.rotations[1] @ res_ns.rotations[0].T
    assert np.allclose(Rrel_ns, Rrel, atol=1e-8)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        generalized_procrustes([np.ones((5, 2))])
    with pytest.raises(ValueError):
        generalized_procrustes([np.ones((5, 2)), np.ones((5, 2))])
    with pytest.raises(ValueError):
        generalized_procrustes([np.zeros((5, 2)), np.zeros((4, 2))])
