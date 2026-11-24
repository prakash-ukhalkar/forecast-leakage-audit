"""Tests for the interleaved-validation leakage ceiling."""
import numpy as np
import pytest

from sdg_leakage_audit.audit.theory import (
    interpolation_variance,
    leakage_ceiling,
    ma_weights,
    prediction_variance,
)


@pytest.mark.parametrize("phi", [0.3, 0.6, 0.9, 0.98])
def test_ar1_closed_forms(phi):
    assert interpolation_variance(np.array([phi])) == pytest.approx(1 / (1 + phi**2), rel=2e-3)
    for h in (1, 6, 24):
        assert prediction_variance(np.array([phi]), h) == pytest.approx((1 - phi ** (2 * h)) / (1 - phi**2), rel=1e-9)
        expected = 1 - (1 - phi**2) / ((1 + phi**2) * (1 - phi ** (2 * h)))
        assert leakage_ceiling([phi], h, metric="mse") == pytest.approx(expected, abs=2e-3)


def test_ceiling_increases_with_persistence_and_horizon_and_vanishes_for_white_noise():
    assert leakage_ceiling([0.0], 6) == pytest.approx(0.0, abs=1e-3)
    assert leakage_ceiling([0.9], 6) > leakage_ceiling([0.5], 6) > leakage_ceiling([0.2], 6)
    assert leakage_ceiling([0.9], 24) > leakage_ceiling([0.9], 6) > leakage_ceiling([0.9], 1)


def test_ma_weights_ar2():
    psi = ma_weights(np.array([0.5, 0.3]), 4)
    assert psi[0] == 1 and psi[1] == pytest.approx(0.5) and psi[2] == pytest.approx(0.5 * 0.5 + 0.3)
