"""Tests for pySC.tuning.c_minus: coupling (c_minus) measurement and correction."""
import pytest
import numpy as np
import at

from pySC.tuning.c_minus import CMinus


pytestmark = pytest.mark.slow


@pytest.fixture
def sc_with_coupling(sc_tuning):
    """SC with skew quadrupole controls for coupling correction.

    Uses sextupole A1 (skew quad) components as coupling correctors.

    Note: with the HMBA single-cell fixture, knob creation can fail because the
    c_minus response matrix is not SVD-invertible.
    """
    sc = sc_tuning
    ring = sc.lattice.design
    sext_names = [ring[i].FamName for i, e in enumerate(ring) if isinstance(e, at.Sextupole)]

    # A1 components of sextupoles act as skew quads for coupling correction
    skew_controls = [f"{name}/A1" for name in sext_names]
    sc.tuning.c_minus.controls = skew_controls

    # Create c_minus knobs and register them
    knob_data = sc.tuning.c_minus.create_c_minus_knobs()
    for knob_name, knob_control in knob_data.data.items():
        sc.magnet_settings.add_knob(
            knob_name=knob_name,
            control_names=knob_control.control_names,
            weights=knob_control.weights,
        )
        sc.design_magnet_settings.add_knob(
            knob_name=knob_name,
            control_names=knob_control.control_names,
            weights=knob_control.weights,
        )
    sc.magnet_settings.sendall()
    sc.design_magnet_settings.sendall()

    return sc


# The HMBA single-cell c_minus response matrix is numerically unusable for
# pseudoinversion, so these integration tests are expected to fail until a
# better-conditioned fixture is available.

@pytest.mark.xfail(reason="HMBA single-cell c_minus response matrix SVD does not converge", strict=False)
def test_c_minus_response(sc_with_coupling):
    """c_minus_response returns complex array of correct length."""
    sc = sc_with_coupling
    delta_c = sc.tuning.c_minus.c_minus_response(delta=1e-5)
    n_controls = len(sc.tuning.c_minus.controls)
    assert len(delta_c) == n_controls
    assert delta_c.dtype == complex


@pytest.mark.xfail(reason="HMBA single-cell c_minus response matrix SVD does not converge", strict=False)
def test_c_minus_cheat_measurement(sc_with_coupling):
    """cheat() returns a complex c_minus value."""
    sc = sc_with_coupling
    c_minus = sc.tuning.c_minus.cheat()
    assert isinstance(c_minus, (complex, np.complexfloating, float))


@pytest.mark.xfail(reason="HMBA single-cell c_minus response matrix SVD does not converge", strict=False)
def test_c_minus_correct_cheat(sc_with_coupling):
    """Coupling correction with 'cheat' measurement runs without error."""
    sc = sc_with_coupling
    sc.tuning.c_minus.correct(
        measurement_method='cheat',
        n_iter=1,
        gain=0.5,
    )
