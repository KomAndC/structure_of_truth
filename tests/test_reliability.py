import numpy as np

from structure_of_truth.reliability import spearman_brown, split_half_ceiling
from structure_of_truth.seeds import make_rng


def test_spearman_brown_known_values():
    assert spearman_brown(0.5) == 2 * 0.5 / 1.5
    assert spearman_brown(1.0) == 1.0


def _ratings(noise: float, seed: int) -> np.ndarray:
    rng = make_rng(seed)
    pts = rng.normal(size=(15, 2))
    true = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    subj = true[None] + noise * rng.normal(size=(20, 15, 15))
    return (subj + subj.transpose(0, 2, 1)) / 2


def test_ceiling_lower_for_noisier_data():
    clean = split_half_ceiling(_ratings(0.05, 0), seed=1, n_splits=20)
    noisy = split_half_ceiling(_ratings(2.0, 0), seed=1, n_splits=20)
    assert clean.shape == (15,)
    assert clean.mean() > noisy.mean()


def test_ceiling_is_seed_reproducible():
    r = _ratings(0.5, 3)
    assert np.allclose(split_half_ceiling(r, seed=5, n_splits=10), split_half_ceiling(r, seed=5, n_splits=10))
