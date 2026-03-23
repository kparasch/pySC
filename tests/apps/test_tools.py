"""Tests for pySC.apps.tools — orbit averaging utility."""
import numpy as np
import pytest

from pySC.apps.tools import get_average_orbit


class TestGetAverageOrbit:
    def test_get_average_orbit_single_shot(self):
        """With n_orbits=1, std should be zero and mean equals the orbit."""
        orbit_x = np.array([1.0, 2.0, 3.0])
        orbit_y = np.array([4.0, 5.0, 6.0])

        def get_orbit():
            return orbit_x.copy(), orbit_y.copy()

        mean_x, mean_y, std_x, std_y = get_average_orbit(get_orbit, n_orbits=1)

        np.testing.assert_array_almost_equal(mean_x, orbit_x)
        np.testing.assert_array_almost_equal(mean_y, orbit_y)
        np.testing.assert_array_almost_equal(std_x, np.zeros(3))
        np.testing.assert_array_almost_equal(std_y, np.zeros(3))

    def test_get_average_orbit_multiple_shots(self):
        """With n_orbits=10 and deterministic orbits, mean and std are correct."""
        n_bpms = 5
        n_orbits = 10
        rng = np.random.default_rng(42)
        # Pre-generate all orbits so we know the expected mean and std
        all_x = rng.standard_normal((n_bpms, n_orbits))
        all_y = rng.standard_normal((n_bpms, n_orbits))

        call_count = 0

        def get_orbit():
            nonlocal call_count
            idx = call_count
            call_count += 1
            return all_x[:, idx].copy(), all_y[:, idx].copy()

        mean_x, mean_y, std_x, std_y = get_average_orbit(get_orbit, n_orbits=n_orbits)

        expected_mean_x = np.mean(all_x, axis=1)
        expected_mean_y = np.mean(all_y, axis=1)
        expected_std_x = np.std(all_x, axis=1)
        expected_std_y = np.std(all_y, axis=1)

        np.testing.assert_array_almost_equal(mean_x, expected_mean_x)
        np.testing.assert_array_almost_equal(mean_y, expected_mean_y)
        np.testing.assert_array_almost_equal(std_x, expected_std_x)
        np.testing.assert_array_almost_equal(std_y, expected_std_y)

    def test_get_average_orbit_shape(self):
        """Output arrays have the same length as the input orbit."""
        n_bpms = 7

        def get_orbit():
            return np.zeros(n_bpms), np.ones(n_bpms)

        mean_x, mean_y, std_x, std_y = get_average_orbit(get_orbit, n_orbits=3)

        assert len(mean_x) == n_bpms
        assert len(mean_y) == n_bpms
        assert len(std_x) == n_bpms
        assert len(std_y) == n_bpms
