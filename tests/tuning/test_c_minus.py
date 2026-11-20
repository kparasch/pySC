"""Tests for pySC.tuning.c_minus: coupling (c_minus) measurement and correction."""
import pytest
import numpy as np
import at

from pySC.core.control import Control, IndivControl
from pySC.core.magnet import ControlMagnetLink


pytestmark = pytest.mark.slow


@pytest.fixture
def sc_with_coupling(sc_tuning):
    """SC with skew quadrupole controls for coupling correction.

    Adds A2 skew-quadrupole controls to the existing sextupole magnets.
    """
    sc = sc_tuning
    ring = sc.lattice.design
    sext_names = [ring[i].FamName for i, e in enumerate(ring) if isinstance(e, at.Sextupole)]

    skew_controls = [f"{name}/A2" for name in sext_names]
    for settings in (sc.magnet_settings, sc.design_magnet_settings):
        for name, control_name in zip(sext_names, skew_controls):
            magnet = settings.magnets[name]
            settings.add_control(Control(
                name=control_name,
                setpoint=magnet.offset_A[1],
                info=IndivControl(magnet_name=name, component="A", order=2,
                                  is_integrated=False),
            ))
            settings.add_link(ControlMagnetLink(
                link_name=f"{control_name}->{control_name}",
                magnet_name=name, control_name=control_name,
                component="A", order=2,
            ))
            magnet.offset_A[1] = 0.0
        settings.connect_links()
        settings.sendall()
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

def test_c_minus_response(sc_with_coupling):
    """c_minus_response returns complex array of correct length."""
    sc = sc_with_coupling
    delta_c = sc.tuning.c_minus.c_minus_response(delta=1e-5)
    n_controls = len(sc.tuning.c_minus.controls)
    assert len(delta_c) == n_controls
    assert delta_c.dtype == complex
    assert np.all(np.isfinite(delta_c))
    assert np.linalg.matrix_rank(np.vstack((delta_c.real, delta_c.imag))) == 2


def test_c_minus_cheat_measurement(sc_with_coupling):
    """cheat() returns a complex c_minus value."""
    sc = sc_with_coupling
    c_minus = sc.tuning.c_minus.cheat()
    assert isinstance(c_minus, (complex, np.complexfloating, float))


def test_c_minus_correct_cheat(sc_with_coupling):
    """Skew-quadrupole correction reduces deliberately introduced coupling."""
    sc = sc_with_coupling
    sc.tuning.c_minus.trim(real=1e-3, imag=5e-4)
    before = abs(sc.tuning.c_minus.cheat())
    assert before > 1e-6
    sc.tuning.c_minus.correct(
        measurement_method='cheat',
        n_iter=1,
        gain=0.5,
    )
    assert abs(sc.tuning.c_minus.cheat()) < before
