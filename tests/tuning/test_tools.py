"""Tests for pySC.tuning.tools: tuning utility functions."""
import pytest
import numpy as np
import warnings

from pySC.tuning.tools import get_average_orbit


pytestmark = pytest.mark.slow


def test_get_average_orbit_delegates_to_apps(sc_tuning):
    """get_average_orbit from tuning.tools delegates to apps.tools with deprecation warning."""
    sc = sc_tuning

    def get_orbit():
        return sc.bpm_system.capture_orbit()

    # The tuning.tools version emits a deprecation warning via logger.warning,
    # but we just verify it returns the expected tuple of 4 arrays.
    result = get_average_orbit(get_orbit=get_orbit, n_orbits=3)
    assert len(result) == 4
    mean_x, mean_y, std_x, std_y = result
    n_bpms = len(sc.bpm_system.names)
    assert mean_x.shape == (n_bpms,)
    assert mean_y.shape == (n_bpms,)
    assert std_x.shape == (n_bpms,)
    assert std_y.shape == (n_bpms,)


def test_get_average_orbit_single_orbit(sc_tuning):
    """With n_orbits=1, std is zero."""
    sc = sc_tuning

    def get_orbit():
        return sc.bpm_system.capture_orbit()

    mean_x, mean_y, std_x, std_y = get_average_orbit(get_orbit=get_orbit, n_orbits=1)
    np.testing.assert_array_equal(std_x, 0.0)
    np.testing.assert_array_equal(std_y, 0.0)
