"""Tests for pySC.tuning.tune: tune measurement and correction."""
import pytest
import numpy as np

from pySC.tuning.tune import Tune


pytestmark = pytest.mark.slow


@pytest.fixture
def sc_with_tune_knobs(sc_tuning):
    """SC with tune correction knobs configured.

    Uses two families of quadrupoles as tune knobs:
    - controls_1: focusing quads (names starting with 'QF')
    - controls_2: defocusing quads (names starting with 'QD')
    """
    sc = sc_tuning
    ring = sc.lattice.design
    import at

    quad_names = [ring[i].FamName for i, e in enumerate(ring) if isinstance(e, at.Quadrupole)]
    focusing = [f"{name}/B2" for name in quad_names if name.startswith("QF")]
    defocusing = [f"{name}/B2" for name in quad_names if name.startswith("QD")]

    sc.tuning.tune.controls_1 = focusing
    sc.tuning.tune.controls_2 = defocusing

    # Create tune knobs and register them
    knob_data = sc.tuning.tune.create_tune_knobs()
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


@pytest.mark.xfail(reason="HMBA single-cell ring is unstable for tune computation")
def test_tune_response_matrix(sc_with_tune_knobs):
    """build_tune_response_matrix returns a 2x2 matrix."""
    sc = sc_with_tune_knobs
    TRM = sc.tuning.tune.build_tune_response_matrix()
    assert TRM.shape == (2, 2)
    # Should have non-zero elements
    assert np.all(np.abs(TRM) > 0)


@pytest.mark.xfail(reason="HMBA single-cell ring is unstable for tune computation")
def test_tune_trim(sc_with_tune_knobs):
    """Tune trim changes the setpoints of knobs without error."""
    sc = sc_with_tune_knobs
    # Apply a small tune shift
    sc.tuning.tune.trim(dqx=0.001, dqy=-0.001)
    # Verify knob setpoint was changed
    qx_sp = sc.magnet_settings.get(sc.tuning.tune.knob_qx)
    qy_sp = sc.magnet_settings.get(sc.tuning.tune.knob_qy)
    assert qx_sp == pytest.approx(0.001)
    assert qy_sp == pytest.approx(-0.001)


@pytest.mark.xfail(reason="HMBA single-cell ring is unstable for tune computation")
def test_tune_correct_cheat(sc_with_tune_knobs):
    """Tune correction with 'cheat' measurement runs without error."""
    sc = sc_with_tune_knobs
    sc.tuning.tune.correct(
        measurement_method='cheat',
        n_iter=1,
        gain=0.5,
    )
