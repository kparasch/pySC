"""Tests for pySC.tuning.averaging: orbit averaging utilities."""
import pytest
import numpy as np

from pySC.tuning.averaging import get_average_orbit


pytestmark = pytest.mark.slow


def test_tuning_get_average_orbit(sc_tuning):
    """get_average_orbit returns (x_avg, y_avg, x_std, y_std) with correct shapes."""
    sc = sc_tuning
    n_bpms = len(sc.bpm_system.names)

    x_avg, y_avg, x_std, y_std = get_average_orbit(sc, n_shots=5)

    assert x_avg.shape == (n_bpms,)
    assert y_avg.shape == (n_bpms,)
    assert x_std.shape == (n_bpms,)
    assert y_std.shape == (n_bpms,)

    # Averages should be finite
    assert np.all(np.isfinite(x_avg))
    assert np.all(np.isfinite(y_avg))
    # Std should be non-negative
    assert np.all(x_std >= 0)
    assert np.all(y_std >= 0)


def test_tuning_get_average_orbit_single_shot(sc_tuning):
    """With n_shots=1, std is zero."""
    sc = sc_tuning

    x_avg, y_avg, x_std, y_std = get_average_orbit(sc, n_shots=1)

    np.testing.assert_array_equal(x_std, 0.0)
    np.testing.assert_array_equal(y_std, 0.0)
