import numpy as np
import pytest

from structure_of_truth.reliability import (
    dataset_noise_ceiling,
    spearman_brown,
    split_half_per_concept,
)
from structure_of_truth.seeds import make_rng


def test_spearman_brown_known_values():
    assert spearman_brown(0.5) == pytest.approx(2 / 3)
    assert spearman_brown(1.0) == 1.0
    assert spearman_brown(0.0) == 0.0


def _ratings(noise: float, seed: int, K: int = 20, n: int = 15) -> np.ndarray:
    rng = make_rng(seed)
    pts = rng.normal(size=(n, 3))
    true = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    E = noise * rng.normal(size=(K, n, n))
    subj = true[None] + (E + E.transpose(0, 2, 1)) / 2
    for s in subj:
        np.fill_diagonal(s, 0)
    return subj


def test_noise_free_ceiling_is_one():
    r = _ratings(0.0, 0)
    c = split_half_per_concept(r, seed=1, n_splits=20)
    assert np.allclose(c.ceiling, 1.0)
    d = dataset_noise_ceiling(r, seed=1, n_splits=20)
    assert d.split_half == pytest.approx(1.0) and d.lower == pytest.approx(1.0) and d.upper == pytest.approx(1.0)


def test_default_is_200_splits_and_shapes():
    c = split_half_per_concept(_ratings(0.5, 0), seed=1)
    assert c.n_splits == 200 and c.split_r.shape == (200, 15) and c.ceiling.shape == (15,)
    assert np.allclose(c.ceiling, spearman_brown(c.split_half_r))


def test_ceiling_lower_for_noisier_data():
    clean = split_half_per_concept(_ratings(0.1, 0), seed=1, n_splits=50).ceiling
    noisy = split_half_per_concept(_ratings(2.0, 0), seed=1, n_splits=50).ceiling
    assert clean.mean() > noisy.mean()
    dc = dataset_noise_ceiling(_ratings(0.1, 0), seed=1, n_splits=50)
    dn = dataset_noise_ceiling(_ratings(2.0, 0), seed=1, n_splits=50)
    assert dc.split_half > dn.split_half
    assert dn.lower <= dn.upper


def test_concept_specific_noise_is_localised():
    r = _ratings(0.2, 3)
    rng = make_rng(4)
    r[:, 0, 1:] += 5 * rng.normal(size=(20, 14))  # concept 0 judged erratically
    r[:, 1:, 0] = r[:, 0, 1:]
    c = split_half_per_concept(r, seed=2, n_splits=50).ceiling
    assert c[0] < c[1:].min()


def test_invariant_to_scale_and_monotone_transform_and_reproducible():
    r = _ratings(0.7, 5)
    base = split_half_per_concept(r, seed=9, n_splits=30).ceiling
    assert np.allclose(split_half_per_concept(10 * r + 2, seed=9, n_splits=30).ceiling, base)
    assert np.array_equal(split_half_per_concept(r, seed=9, n_splits=30).ceiling, base)


def test_handles_missing_values():
    r = _ratings(0.3, 6)
    rng = make_rng(7)
    mask = rng.random(r.shape) < 0.2
    r[mask] = np.nan
    c = split_half_per_concept(r, seed=1, n_splits=10)
    assert np.all(np.isfinite(c.ceiling))
    assert np.isfinite(dataset_noise_ceiling(r, seed=1, n_splits=10).split_half)
