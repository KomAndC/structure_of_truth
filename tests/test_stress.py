import numpy as np
import pytest

from conftest import pairwise, random_orthogonal
from structure_of_truth.seeds import make_rng
from structure_of_truth.stress import stress, stress_from_config


def _config(seed=0, n=12):
    return make_rng(seed).normal(size=(n, 2))


@pytest.mark.parametrize("level", ["raw", "ratio", "interval", "ordinal"])
def test_perfect_fit_is_zero_everywhere(level):
    X = _config()
    r = stress_from_config(pairwise(X), X, level=level)
    assert r.stress1 == pytest.approx(0, abs=1e-12)
    assert np.allclose(r.pair, 0) and np.allclose(r.concept, 0)


def test_decomposition_sums_to_stress_squared():
    X = _config()
    rng = make_rng(1)
    delta = pairwise(X + 0.3 * rng.normal(size=X.shape))
    r = stress_from_config(delta, X)
    iu = np.triu_indices(len(X), 1)
    assert r.pair[iu].sum() == pytest.approx(r.stress1**2)
    assert r.concept.sum() == pytest.approx(r.stress1**2)
    assert np.allclose(r.pair, r.pair.T) and np.allclose(np.diag(r.pair), 0)


def test_invariant_to_rotation_translation_and_scale():
    X = _config()
    delta = pairwise(X + 0.3 * make_rng(2).normal(size=X.shape))
    a = stress_from_config(delta, X)
    b = stress_from_config(5 * delta, 0.1 * X @ random_orthogonal(2, 3) + 7)
    assert b.stress1 == pytest.approx(a.stress1)
    assert np.allclose(b.concept, a.concept)


def test_ordinal_ignores_monotone_transform():
    X = _config()
    delta = np.exp(pairwise(X))
    np.fill_diagonal(delta, 0)
    assert stress_from_config(delta, X, level="ordinal").stress1 == pytest.approx(0, abs=1e-12)
    assert stress_from_config(delta, X, level="ratio").stress1 > 0.05


def test_local_deformation_concentrates_stress():
    X = _config(n=15)
    moved = X.copy()
    moved[0] += [3.0, 0.0]  # one concept misplaced in the data
    r = stress_from_config(pairwise(moved), X, level="raw")
    assert r.concept.argmax() == 0
    assert r.concept_stress1.argmax() == 0
    j = np.unravel_index(r.pair.argmax(), r.pair.shape)
    assert 0 in j


def test_missing_pairs_excluded():
    X = _config()
    delta = pairwise(X)
    delta[0, 1] = delta[1, 0] = np.nan
    r = stress(delta, pairwise(X))
    assert r.stress1 == pytest.approx(0, abs=1e-12) and r.pair[0, 1] == 0
