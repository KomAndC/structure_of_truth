import numpy as np
import pytest

import propaganda
from structure_of_truth.seeds import make_rng


@pytest.mark.parametrize("n_c", [3, 20, 80])
def test_rigid_transform_hits_target_rms_and_preserves_cluster_distances(n_c):
    rng = make_rng(n_c)
    X, layer, D0, md = propaganda.world(rng)
    C = propaganda.pick_cluster(rng, D0, layer, n_c)
    T, th, tl = propaganda.rigid_transform(rng, X[C], 0.2 * md)
    Y = T(X[C])
    rms = np.sqrt(((Y - X[C]) ** 2).sum(1).mean())
    assert rms == pytest.approx(0.2 * md, rel=1e-6)
    assert 0 <= th <= np.pi / 2 + 1e-12
    assert np.allclose(np.linalg.norm(Y[:, None] - Y[None], axis=-1), D0[np.ix_(C, C)])


def test_cluster_is_nearest_neighbours_of_a_value_laden_seed():
    rng = make_rng(1)
    X, layer, D0, md = propaganda.world(rng)
    C = propaganda.pick_cluster(rng, D0, layer, 10)
    assert layer[C[0]] == 2 and len(set(C)) == 10
    assert D0[C[0], C].max() <= np.sort(D0[C[0]])[9] + 1e-12


def test_noise_free_rigid_signature_and_controls():
    rng = make_rng(2)
    X, layer, D0, md = propaganda.world(rng)
    C = propaganda.pick_cluster(rng, D0, layer, 20)
    knn = propaganda.knn_mask(D0)
    T, _, _ = propaganda.rigid_transform(rng, X[C], 0.2 * md)
    Y = X.copy(); Y[C] = T(X[C])
    rig = propaganda.interface_stats(np.linalg.norm(Y[:, None] - Y[None], axis=-1), X, D0, C, knn)
    Y2 = X.copy(); Y2[C] += propaganda.random_displacement(rng, 20, 5, 0.2 * md)
    non = propaganda.interface_stats(np.linalg.norm(Y2[:, None] - Y2[None], axis=-1), X, D0, C, knn)
    assert rig["auc_cr_cc"] > 0.9 and non["auc_cr_cc"] < 0.6
    assert rig["rigidity"] < 0.05 < non["rigidity"]


def test_consistency_tension_zero_at_both_ends():
    for q in (0.0, 1.0):
        r = propaganda.run_consistency((1, (0, 0, 0, 0, 0), "rigid", 10, 0.0, 0.2, q, 0))
        assert r["stress1"] < 1e-3
    mid = propaganda.run_consistency((1, (0, 0, 0, 0, 0), "rigid", 10, 0.0, 0.2, 0.5, 0))
    assert mid["stress1"] > 5e-3
