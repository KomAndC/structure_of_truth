import numpy as np
import pytest

from generator import SimConfig, make_population, measure, true_magnitudes
from structure_of_truth.seeds import make_rng


def _pop(gen, noise=0.0, seed=0, S=5):
    rng = make_rng(seed)
    pop = make_population(SimConfig(n_subjects=S, generator=gen, n_per_layer=10, k=3, noise=noise), rng)
    return pop, rng


def test_shared_space_is_whitened():
    pop, _ = _pop("null")
    N = pop.X.shape[0]
    assert np.allclose(pop.X.T @ pop.X / N, np.eye(3))


def test_null_subjects_identical_without_noise():
    pop, rng = _pop("null")
    D, D_true = measure(pop, rng)
    assert np.allclose(D_true, D_true[0][None])
    assert np.allclose(true_magnitudes(pop)[0], 0) and np.allclose(true_magnitudes(pop)[1], 0)


def test_rotation_changes_only_value_anchor_distances():
    pop, rng = _pop("A")
    _, D_true = measure(pop, rng)
    v, a = pop.value_idx, pop.anchor_idx
    D0 = np.linalg.norm(pop.X[:, None] - pop.X[None], axis=-1)
    for D in D_true:
        assert np.allclose(D[np.ix_(v, v)], D0[np.ix_(v, v)])
        assert np.allclose(D[np.ix_(a, a)], D0[np.ix_(a, a)])
    assert not np.allclose(D_true[0][np.ix_(v, a)], D0[np.ix_(v, a)])


def test_rescaling_has_unit_geometric_mean_weights():
    pop, _ = _pop("B")
    assert np.allclose(pop.log_w.sum(axis=1), 0)
    assert np.allclose(pop.theta, 0)


def test_noise_is_layer_specific_and_reproducible():
    pop, rng = _pop("null", noise=0.5, S=200)
    D, D_true = measure(pop, rng)
    E = D - D_true
    l = pop.layer
    sd = [E[:, np.ix_(l == L, l == L)[0], np.ix_(l == L, l == L)[1]].std() for L in range(3)]
    assert sd[0] < sd[1] < sd[2]
    pop2, rng2 = _pop("null", noise=0.5, S=200)
    assert np.array_equal(measure(pop2, rng2)[0], D)


def test_bad_generator_rejected():
    with pytest.raises(ValueError):
        SimConfig(generator="X")
