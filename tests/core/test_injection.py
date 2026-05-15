"""Tests for pySC.core.injection: InjectionSettings."""
import pytest
import numpy as np
from unittest.mock import MagicMock

from pySC.core.injection import InjectionSettings
from pySC.core.rng import RNG


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_injection(n_particles=1, seed=42, **kwargs):
    """Return an InjectionSettings wired to a mock SC with a real RNG."""
    inj = InjectionSettings(n_particles=n_particles, **kwargs)
    mock_sc = MagicMock()
    mock_sc.rng = RNG(seed=seed)
    inj._parent = mock_sc
    return inj


# ---------------------------------------------------------------------------
# generate_bunch
# ---------------------------------------------------------------------------

def test_generate_bunch_single_particle():
    """With n_particles=1, bunch shape is (1, 6)."""
    inj = _make_injection(n_particles=1)
    bunch = inj.generate_bunch(use_design=True)
    assert bunch.shape == (1, 6)


def test_generate_bunch_design_uses_nominal():
    """generate_bunch(use_design=True) uses the nominal x, not x_inj."""
    inj = _make_injection(
        n_particles=1,
        x=0.001, px=0.0, y=0.002, py=0.0, tau=0.0, delta=0.0,
        x_error_syst=0.1, y_error_syst=0.1,  # large systematic errors
    )
    bunch = inj.generate_bunch(use_design=True)
    # With use_design=True, x coordinate should be exactly x (no injection errors)
    assert bunch[0, 0] == pytest.approx(0.001)
    assert bunch[0, 2] == pytest.approx(0.002)


def test_generate_bunch_error_uses_rng():
    """generate_bunch(use_design=False) adds random injection errors via RNG."""
    inj = _make_injection(
        n_particles=1, seed=42,
        x=0.0, px=0.0, y=0.0, py=0.0, tau=0.0, delta=0.0,
        x_error_syst=0.5, x_error_stat=0.1,
    )
    bunch = inj.generate_bunch(use_design=False)
    # With errors, the x coordinate should not be exactly zero
    assert bunch[0, 0] != pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# generate_zero_centered_bunch
# ---------------------------------------------------------------------------

def test_generate_zero_centered_single_particle():
    """With n_particles=1, returns zeros (no distribution)."""
    inj = _make_injection(n_particles=1)
    bunch = inj.generate_zero_centered_bunch()
    assert bunch.shape == (1, 6)
    np.testing.assert_array_equal(bunch, np.zeros((1, 6)))


def test_generate_zero_centered_multi_particle():
    """With n_particles>1, distribution uses Courant-Snyder parameterization.

    Verifies the covariance structure: the x-px correlation reflects alfx,
    and the y-py correlation reflects alfy (not alfx — Bug 1 was a copy-paste
    error using alfx for the vertical plane).
    """
    alfx = 1.0
    alfy = -0.5
    betx = 10.0
    bety = 5.0
    inj = _make_injection(
        n_particles=10000, seed=99,
        betx=betx, alfx=alfx, bety=bety, alfy=alfy,
        gemit_x=1e-9, gemit_y=1e-9,
        bunch_length=0.01, energy_spread=1e-3,
    )
    bunch = inj.generate_zero_centered_bunch()
    assert bunch.shape == (10000, 6)
    # The bunch should be zero-centred on average (mean close to zero)
    for col in range(6):
        assert np.abs(np.mean(bunch[:, col])) < 5 * np.std(bunch[:, col]) / np.sqrt(10000)

    # Check covariance structure: <x*px> / <x^2> = -alfx/betx
    # and <y*py> / <y^2> = -alfy/bety
    cov = np.cov(bunch.T)
    x_px_ratio = cov[0, 1] / cov[0, 0]
    y_py_ratio = cov[2, 3] / cov[2, 2]
    np.testing.assert_allclose(x_px_ratio, -alfx / betx, atol=0.05)
    np.testing.assert_allclose(y_py_ratio, -alfy / bety, atol=0.05)


# ---------------------------------------------------------------------------
# generate_orbit_centered_bunch
# ---------------------------------------------------------------------------

def test_generate_orbit_centered_bunch():
    """Bunch is centered on the closed orbit."""
    inj = _make_injection(n_particles=1, seed=42)
    # Mock twiss to return a known closed orbit
    twiss = {
        "x": np.array([0.001]),
        "px": np.array([0.0002]),
        "y": np.array([-0.0005]),
        "py": np.array([0.0001]),
        "tau": np.array([0.0]),
        "delta": np.array([0.0]),
    }
    inj._parent.lattice.get_twiss.return_value = twiss

    bunch = inj.generate_orbit_centered_bunch(use_design=True)
    assert bunch[0, 0] == pytest.approx(0.001)
    assert bunch[0, 1] == pytest.approx(0.0002)
    assert bunch[0, 2] == pytest.approx(-0.0005)


# ---------------------------------------------------------------------------
# _inj properties
# ---------------------------------------------------------------------------

def test_injection_properties_add_errors():
    """x_inj = x + normal(x_error_syst, x_error_stat)."""
    inj = _make_injection(
        n_particles=1, seed=42,
        x=0.001,
        x_error_syst=0.01,  # systematic (mean of random draw)
        x_error_stat=0.001,  # statistical (std of random draw)
    )
    # x_inj calls rng.normal(loc=x_error_syst, scale=x_error_stat) + x
    val = inj.x_inj
    # It should not be exactly x (unless rng.normal returns exactly -x_error_syst, very unlikely)
    assert val != pytest.approx(inj.x, abs=1e-15)
    # But it should be in a reasonable range around x + x_error_syst
    assert abs(val - (inj.x + inj.x_error_syst)) < 10 * inj.x_error_stat


# ---------------------------------------------------------------------------
# REGRESSION: invW missing return statement
# ---------------------------------------------------------------------------

@pytest.mark.regression
def test_invW_returns_courant_snyder_matrix():
    """InjectionSettings.invW returns the correct 6x6 Courant-Snyder matrix.

    Regression: previously the property had no return statement and always
    returned None.
    """
    betx, alfx = 10.0, 1.0
    bety, alfy = 5.0, -0.5
    inj = _make_injection(betx=betx, alfx=alfx, bety=bety, alfy=alfy)
    result = inj.invW
    assert result is not None, "invW should return a matrix, not None"
    assert result.shape == (6, 6)

    # Verify Courant-Snyder parameterization
    sbetx = betx**0.5
    sbety = bety**0.5
    assert result[0, 0] == pytest.approx(sbetx)
    assert result[1, 0] == pytest.approx(-alfx / sbetx)
    assert result[1, 1] == pytest.approx(1 / sbetx)
    assert result[2, 2] == pytest.approx(sbety)
    assert result[3, 2] == pytest.approx(-alfy / sbety)
    assert result[3, 3] == pytest.approx(1 / sbety)
    assert result[4, 4] == pytest.approx(1)
    assert result[5, 5] == pytest.approx(1)
