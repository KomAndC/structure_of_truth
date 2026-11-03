import numpy as np
import pytest

import power
from structure_of_truth.seeds import make_rng


@pytest.mark.parametrize("m", [10, 40, 179])
def test_pair_design_is_m_regular_without_duplicates(m):
    pairs, nbr = power.pair_design(180, m, make_rng(0))
    assert nbr.shape == (180, m)
    assert len({tuple(p) for p in pairs}) == len(pairs) == 180 * m // 2
    assert np.all(pairs[:, 0] < pairs[:, 1])
    for i in (0, 77):
        assert np.all((pairs[nbr[i]] == i).any(axis=1))


def _lowrank(S=80, P=600, r=4, noise=0.3, seed=1):
    rng = make_rng(seed)
    L = rng.normal(size=(S, r)) @ (3 * rng.normal(size=(r, P)))
    R1 = L + noise * rng.normal(size=(S, P))
    R2 = L + noise * rng.normal(size=(S, P))
    return R1 - R1.mean(0), R2 - R2.mean(0), rng


def test_rank_methods_recover_clean_low_rank():
    R1, R2, rng = _lowrank()
    lam = power.gram_eig(R1)[0]
    assert power.rank_parallel(R1, lam, rng) == 4
    assert power.rank_cv(R1, rng) == 4
    assert power.rank_retest(R1, R2, rng) == 4
    assert power.rank_scree(lam) == 4
    p, share = power.reliable_share(R1, R2, 4)
    assert p < 1e-6 and share > 0.95


def test_pure_noise_gives_rank_zero_and_no_reliable_variance():
    rng = make_rng(2)
    R1 = rng.normal(size=(60, 500))
    R2 = rng.normal(size=(60, 500))
    R1, R2 = R1 - R1.mean(0), R2 - R2.mean(0)
    lam = power.gram_eig(R1)[0]
    assert power.rank_parallel(R1, lam, rng) == 0
    assert power.rank_cv(R1, rng) == 0
    assert power.reliable_share(R1, R2, 0)[0] > 0.01


def test_rowcorr_matches_numpy():
    rng = make_rng(3)
    a, b = rng.normal(size=(5, 30)), rng.normal(size=(5, 30))
    assert np.allclose(power.rowcorr(a, b), [np.corrcoef(x, y)[0, 1] for x, y in zip(a, b)])


def test_profile_reliability_matches_library_on_full_design():
    from structure_of_truth.reliability import split_half_per_concept

    rng = make_rng(4)
    X, layer, md = power.base_world(rng)
    X, layer = X[:30], layer[:30]
    pairs, nbr = power.pair_design(30, 29, rng)
    V = power.measure_pairs(X[None] + 0.3 * rng.normal(size=(40, 30, 5)), pairs, 0.5, rng)
    full = np.zeros((40, 30, 30))
    full[:, pairs[:, 0], pairs[:, 1]] = V
    full = full + full.transpose(0, 2, 1)
    ours = power.profile_reliability(V, nbr, make_rng(9), 200)
    lib = split_half_per_concept(full, seed=9, n_splits=200, method="pearson").ceiling
    assert np.allclose(ours, lib, atol=0.02)
