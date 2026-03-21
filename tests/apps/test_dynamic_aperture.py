"""Tests for pySC.apps.dynamic_aperture — DA scan and helpers."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pySC.apps.dynamic_aperture import (
    dynamic_aperture,
    _is_lost,
    _shoelace_area,
)


class TestShoelaceArea:
    """Test _shoelace_area polygon area computation."""

    def test_unit_square(self):
        x = np.array([0, 1, 1, 0])
        y = np.array([0, 0, 1, 1])
        assert _shoelace_area(x, y) == pytest.approx(1.0)

    def test_triangle(self):
        x = np.array([0, 2, 0])
        y = np.array([0, 0, 3])
        assert _shoelace_area(x, y) == pytest.approx(3.0)

    def test_degenerate_line(self):
        x = np.array([0, 1, 2])
        y = np.array([0, 0, 0])
        assert _shoelace_area(x, y) == pytest.approx(0.0)

    def test_circle_approximation(self):
        """Regular n-gon area approaches pi*r^2 for large n."""
        n = 1000
        r = 1.0
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        np.testing.assert_allclose(_shoelace_area(x, y), np.pi * r**2, rtol=1e-4)


class TestIsLost:
    """Test _is_lost particle tracking wrapper."""

    def test_stable_particle(self):
        SC = MagicMock()
        SC.lattice.track.return_value = np.zeros((1, 6))
        assert not _is_lost(SC, radius=1e-3, theta=0, x0=0, y0=0, n_turns=100)

    def test_lost_particle(self):
        SC = MagicMock()
        result = np.full((1, 6), np.nan)
        SC.lattice.track.return_value = result
        assert _is_lost(SC, radius=0.1, theta=0, x0=0, y0=0, n_turns=100)

    def test_initial_conditions_include_offset(self):
        SC = MagicMock()
        SC.lattice.track.return_value = np.zeros((1, 6))

        _is_lost(SC, radius=1e-3, theta=np.pi / 4, x0=0.001, y0=-0.002, n_turns=10)

        bunch = SC.lattice.track.call_args.args[0]
        expected_x = 1e-3 * np.cos(np.pi / 4) + 0.001
        expected_y = 1e-3 * np.sin(np.pi / 4) - 0.002
        np.testing.assert_allclose(bunch[0, 0], expected_x)
        np.testing.assert_allclose(bunch[0, 2], expected_y)


class TestDynamicAperture:
    """Test dynamic_aperture scan with mocked tracking."""

    def _make_sc(self, loss_radius=0.01):
        """SC mock where particles are lost beyond loss_radius."""
        SC = MagicMock()
        SC.lattice.get_orbit.return_value = np.zeros((2, 1))

        def track_fn(bunch, n_turns=1000):
            r = np.sqrt(bunch[0, 0] ** 2 + bunch[0, 2] ** 2)
            if r > loss_radius:
                return np.full_like(bunch, np.nan)
            return bunch

        SC.lattice.track.side_effect = track_fn
        return SC

    def test_returns_expected_keys(self):
        SC = self._make_sc()
        result = dynamic_aperture(SC, n_angles=4, n_turns=10, accuracy=1e-4)
        for key in ("radii", "angles", "area", "x", "y"):
            assert key in result

    def test_radii_shape_matches_angles(self):
        n = 8
        SC = self._make_sc()
        result = dynamic_aperture(SC, n_angles=n, n_turns=10, accuracy=1e-4)
        assert result["radii"].shape == (n,)
        assert result["angles"].shape == (n,)

    def test_radii_close_to_loss_radius(self):
        loss_r = 0.01
        SC = self._make_sc(loss_radius=loss_r)
        result = dynamic_aperture(
            SC, n_angles=4, n_turns=10,
            accuracy=1e-5, initial_radius=1e-3,
        )
        # All radii should converge near loss_radius
        np.testing.assert_allclose(result["radii"], loss_r, atol=1e-4)

    def test_area_positive(self):
        SC = self._make_sc(loss_radius=0.01)
        result = dynamic_aperture(SC, n_angles=16, n_turns=10, accuracy=1e-5)
        assert result["area"] > 0

    def test_zero_aperture_when_always_lost(self):
        """If particles are lost at any radius, DA is zero."""
        SC = MagicMock()
        SC.lattice.get_orbit.return_value = np.zeros((2, 1))
        SC.lattice.track.return_value = np.full((1, 6), np.nan)

        result = dynamic_aperture(SC, n_angles=4, n_turns=10, accuracy=1e-4)
        np.testing.assert_array_equal(result["radii"], 0)
        assert result["area"] == pytest.approx(0.0)

    def test_center_on_orbit_false(self):
        """With center_on_orbit=False, orbit is not queried."""
        SC = self._make_sc(loss_radius=0.005)
        result = dynamic_aperture(
            SC, n_angles=4, n_turns=10,
            accuracy=1e-5, center_on_orbit=False,
        )
        SC.lattice.get_orbit.assert_not_called()
