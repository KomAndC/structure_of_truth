import numpy as np
from scipy.stats import special_ortho_group

from structure_of_truth.procrustes import generalized_procrustes
from structure_of_truth.seeds import make_rng


def test_recovers_shared_space_under_pure_rotation():
    rng = make_rng(0)
    base = rng.normal(size=(30, 3))
    rots = special_ortho_group.rvs(3, size=5, random_state=1)
    configs = np.stack([base @ R for R in rots])
    res = generalized_procrustes(configs)
    assert res.residual_ss < 1e-12


def test_residual_grows_with_noise():
    rng = make_rng(0)
    base = rng.normal(size=(30, 3))
    low = np.stack([base + 0.01 * rng.normal(size=base.shape) for _ in range(5)])
    high = np.stack([base + 0.5 * rng.normal(size=base.shape) for _ in range(5)])
    assert generalized_procrustes(low).residual_ss < generalized_procrustes(high).residual_ss
