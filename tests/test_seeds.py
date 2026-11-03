import numpy as np
import pytest

from structure_of_truth.seeds import child_seeds, make_rng


def test_same_seed_same_stream():
    assert np.array_equal(make_rng(7).normal(size=5), make_rng(7).normal(size=5))


def test_seed_must_be_int():
    with pytest.raises(TypeError):
        make_rng(None)


def test_child_seeds_reproducible_and_distinct():
    a, b = child_seeds(1, 4), child_seeds(1, 4)
    assert a == b and len(set(a)) == 4
