import numpy as np
import pytest
from scipy.linalg import expm

from conftest import pairwise, random_orthogonal
from structure_of_truth.decompose import (
    block_rotation,
    classical_mds,
    estimate_components,
    rescaling_magnitudes,
    rotation_magnitude,
    similarity_align,
    weighted_block_rotation,
)
from structure_of_truth.seeds import make_rng


def test_classical_mds_reproduces_distances():
    X = make_rng(0).normal(size=(20, 3))
    assert np.allclose(pairwise(classical_mds(pairwise(X), 3)), pairwise(X))


def test_similarity_align_on_anchor_rows():
    rng = make_rng(1)
    X = rng.normal(size=(20, 3))
    Y = 2.0 * X @ random_orthogonal(3, 2, reflect=True) + 5
    Y[15:] += 10 * rng.normal(size=(5, 3))  # non-anchor rows disturbed
    Z = similarity_align(Y, X, np.arange(15))
    assert np.allclose(Z[:15], X[:15])


def _skew(rng, k):
    M = rng.normal(size=(k, k))
    O = M - M.T
    return O / np.linalg.norm(O) * np.sqrt(2)


@pytest.mark.parametrize("weighted", [False, True])
def test_block_rotation_recovers_known_rotation(weighted):
    rng = make_rng(3)
    X = rng.normal(size=(30, 4))
    R = expm(0.4 * _skew(rng, 4))
    d = np.exp(0.3 * rng.normal(size=4)) if weighted else np.ones(4)
    Y = (X @ R) * d + 1.0
    Rhat = weighted_block_rotation(X, Y, np.arange(30), d) if weighted else block_rotation(X, Y, np.arange(30))
    assert np.allclose(Rhat, R, atol=1e-6)
    assert rotation_magnitude(Rhat) == pytest.approx(rotation_magnitude(R), abs=1e-6)


def test_rescaling_magnitudes_ignore_subject_size_and_common_axis_weight():
    L = make_rng(4).normal(size=(6, 3))
    shifted = L + 2.0 + np.array([0.5, -1.0, 3.0])[None] + make_rng(5).normal(size=(6, 1))
    assert np.allclose(rescaling_magnitudes(L), rescaling_magnitudes(shifted))
    assert np.allclose(rescaling_magnitudes(np.zeros((4, 3))), 0)


def test_estimate_components_noise_free_matches_truth():
    from generator import SimConfig, make_population, measure, true_magnitudes

    rng = make_rng(6)
    pop = make_population(SimConfig(n_subjects=12, generator="C", n_per_layer=20, k=3, noise=0.0), rng)
    D, _ = measure(pop, rng)
    c = estimate_components(D, 3, block=pop.value_idx, anchors=pop.anchor_idx)
    th, b = true_magnitudes(pop)
    # Not exact even without noise: rotation is identifiable only relative to the group
    # consensus of the value block (INDSCAL has no rotation term, fit ~0.99), not relative
    # to the generator's unrotated X. Rescaling is recovered almost exactly.
    assert np.corrcoef(c.theta_indscal, th)[0, 1] > 0.95
    assert np.corrcoef(c.b_indscal, b)[0, 1] > 0.99
    assert np.abs(c.theta_indscal - th).max() < 0.15
