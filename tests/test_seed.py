"""Tests for src.sdg_leakage_audit.utils.seed."""

from __future__ import annotations

import random

import numpy as np
import pytest

from sdg_leakage_audit.utils.seed import set_global_seed


def test_set_global_seed_reproducible_random_and_numpy():
    set_global_seed(123)
    random_vals_1 = [random.random() for _ in range(5)]
    numpy_vals_1 = np.random.rand(5)

    set_global_seed(123)
    random_vals_2 = [random.random() for _ in range(5)]
    numpy_vals_2 = np.random.rand(5)

    assert random_vals_1 == random_vals_2
    assert np.allclose(numpy_vals_1, numpy_vals_2)


def test_set_global_seed_different_seeds_differ():
    set_global_seed(1)
    numpy_vals_1 = np.random.rand(5)

    set_global_seed(2)
    numpy_vals_2 = np.random.rand(5)

    assert not np.allclose(numpy_vals_1, numpy_vals_2)


def test_set_global_seed_negative_raises():
    with pytest.raises(ValueError):
        set_global_seed(-1)


def test_set_global_seed_torch_if_installed():
    torch = pytest.importorskip("torch")

    set_global_seed(7)
    t1 = torch.rand(5)
    set_global_seed(7)
    t2 = torch.rand(5)

    assert torch.allclose(t1, t2)
