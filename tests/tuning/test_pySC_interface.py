"""Tests for pySC.tuning.pySC_interface: OrbitInterface and InjectionInterface."""
import pytest
import numpy as np

from pySC.tuning.pySC_interface import pySCOrbitInterface, pySCInjectionInterface


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# pySCOrbitInterface
# ---------------------------------------------------------------------------

def test_orbit_interface_get_orbit(sc_tuning):
    """get_orbit returns (x, y) from bpm_system.capture_orbit."""
    interface = pySCOrbitInterface(SC=sc_tuning)
    x, y = interface.get_orbit()
    n_bpms = len(sc_tuning.bpm_system.names)
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(x) == n_bpms
    assert len(y) == n_bpms


def test_orbit_interface_get_set(sc_tuning):
    """get/set delegate to magnet_settings."""
    interface = pySCOrbitInterface(SC=sc_tuning)
    corr = sc_tuning.tuning.HCORR[0]

    # Set a known value
    interface.set(corr, 1.23e-4)
    value = interface.get(corr)
    assert value == pytest.approx(1.23e-4)

    # Reset
    interface.set(corr, 0.0)
    assert interface.get(corr) == pytest.approx(0.0)


def test_orbit_interface_get_set_many(sc_tuning):
    """get_many/set_many delegate correctly."""
    interface = pySCOrbitInterface(SC=sc_tuning)
    names = sc_tuning.tuning.HCORR[:2]
    data = {names[0]: 1.0e-4, names[1]: -2.0e-4}

    interface.set_many(data)
    result = interface.get_many(names)
    assert result[names[0]] == pytest.approx(1.0e-4)
    assert result[names[1]] == pytest.approx(-2.0e-4)

    # Clean up
    interface.set_many({n: 0.0 for n in names})


def test_orbit_interface_rf_frequency(sc_tuning):
    """get_rf_main_frequency/set_rf_main_frequency delegate to RF settings."""
    interface = pySCOrbitInterface(SC=sc_tuning)
    original_freq = interface.get_rf_main_frequency()
    assert original_freq > 0

    # Perturb
    new_freq = original_freq + 100.0
    interface.set_rf_main_frequency(new_freq)
    assert interface.get_rf_main_frequency() == pytest.approx(new_freq)

    # Restore
    interface.set_rf_main_frequency(original_freq)
    assert interface.get_rf_main_frequency() == pytest.approx(original_freq)


def test_orbit_interface_design_mode(sc_tuning):
    """use_design=True reads from design lattice/settings."""
    interface_real = pySCOrbitInterface(SC=sc_tuning, use_design=False)
    interface_design = pySCOrbitInterface(SC=sc_tuning, use_design=True)

    # The design orbit should exist and be finite
    x_d, y_d = interface_design.get_orbit()
    assert np.all(np.isfinite(x_d))
    assert np.all(np.isfinite(y_d))

    # Design RF frequency should match
    freq_design = interface_design.get_rf_main_frequency()
    freq_real = interface_real.get_rf_main_frequency()
    assert freq_design == pytest.approx(freq_real)

    # Design magnet settings should be accessible
    corr = sc_tuning.tuning.HCORR[0]
    val_design = interface_design.get(corr)
    assert isinstance(val_design, float)


# ---------------------------------------------------------------------------
# pySCInjectionInterface
# ---------------------------------------------------------------------------

def test_injection_interface_get_orbit(sc_tuning):
    """get_orbit returns flattened injection trajectory."""
    n_turns = 1
    interface = pySCInjectionInterface(SC=sc_tuning, n_turns=n_turns)
    x, y = interface.get_orbit()
    n_bpms = len(sc_tuning.bpm_system.names)
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    # Flattened: n_bpms * n_turns
    assert len(x) == n_bpms * n_turns
    assert len(y) == n_bpms * n_turns


def test_injection_interface_ref_orbit(sc_tuning):
    """get_ref_orbit tiles reference orbit by n_turns."""
    n_turns = 3
    interface = pySCInjectionInterface(SC=sc_tuning, n_turns=n_turns)
    x_ref, y_ref = interface.get_ref_orbit()
    n_bpms = len(sc_tuning.bpm_system.names)
    assert len(x_ref) == n_bpms * n_turns
    assert len(y_ref) == n_bpms * n_turns
    # Since reference is zero in conftest, all values should be zero
    np.testing.assert_array_equal(x_ref, 0.0)
    np.testing.assert_array_equal(y_ref, 0.0)
