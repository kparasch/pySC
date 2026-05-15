"""Tests for pySC.core.rfsettings: RFCavity, RFSystem, RFSettings."""
import pytest
from unittest.mock import MagicMock

from pySC.core.rfsettings import RFCavity, RFSystem, RFSettings


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_rf_stack(n_cavities=2, voltage=1e6, phase=0.0, frequency=500e6):
    """Build RFSettings -> RFSystem -> RFCavity hierarchy with mock SC parent."""
    rf = RFSettings()
    mock_sc = MagicMock()
    rf._parent = mock_sc

    cavities = {}
    cav_names = []
    for i in range(n_cavities):
        name = f"cav{i}"
        cav = RFCavity(sim_index=i)
        cavities[name] = cav
        cav_names.append(name)

    system = RFSystem(cavities=cav_names, voltage=voltage,
                      phase=phase, frequency=frequency)
    rf.systems["main"] = system
    rf.cavities = cavities

    # Wire parents
    system._parent = rf
    for name in cav_names:
        rf.cavities[name]._parent_system = system

    return rf, system, cavities


# ---------------------------------------------------------------------------
# RFCavity actual_* properties
# ---------------------------------------------------------------------------

def test_rf_cavity_actual_voltage():
    rf, system, cavs = _make_rf_stack(n_cavities=2, voltage=1e6)
    cav = cavs["cav0"]
    cav.voltage_error = 100.0
    cav.voltage_correction = -50.0
    cav.voltage_delta = 25.0

    # system_voltage/n_cavities + error + correction + delta
    expected = 1e6 / 2 + 100 - 50 + 25
    assert cav.actual_voltage == pytest.approx(expected)


def test_rf_cavity_actual_phase():
    rf, system, cavs = _make_rf_stack(phase=180.0)
    cav = cavs["cav0"]
    cav.phase_error = 1.0
    cav.phase_correction = -0.5
    cav.phase_delta = 0.2

    expected = 180.0 + 1.0 - 0.5 + 0.2
    assert cav.actual_phase == pytest.approx(expected)


def test_rf_cavity_actual_frequency():
    rf, system, cavs = _make_rf_stack(frequency=500e6)
    cav = cavs["cav0"]
    cav.frequency_error = 1000.0
    cav.frequency_correction = -500.0
    cav.frequency_delta = 100.0

    expected = 500e6 + 1000 - 500 + 100
    assert cav.actual_frequency == pytest.approx(expected)


# ---------------------------------------------------------------------------
# RFSystem set_* triggers update
# ---------------------------------------------------------------------------

def test_rf_system_set_voltage_triggers_update():
    rf, system, cavs = _make_rf_stack(n_cavities=1, voltage=1e6)
    system.set_voltage(2e6)
    assert system.voltage == pytest.approx(2e6)
    # Verify update was called with the correct arguments
    mock_update = rf._parent.lattice.update_cavity
    mock_update.assert_called_once()
    call_kwargs = mock_update.call_args[1]
    assert call_kwargs["index"] == 0
    assert call_kwargs["voltage"] == pytest.approx(2e6)


def test_rf_system_set_frequency_triggers_update():
    rf, system, cavs = _make_rf_stack(frequency=500e6)
    system.set_frequency(499e6)
    assert system.frequency == pytest.approx(499e6)
    # Verify update was called for each cavity with correct frequency
    mock_update = rf._parent.lattice.update_cavity
    assert mock_update.call_count == 2  # 2 cavities
    for call in mock_update.call_args_list:
        assert call[1]["frequency"] == pytest.approx(499e6)


# ---------------------------------------------------------------------------
# RFSystem.indices
# ---------------------------------------------------------------------------

def test_rf_system_indices_property():
    rf, system, cavs = _make_rf_stack(n_cavities=3)
    assert system.indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# RFSettings.main
# ---------------------------------------------------------------------------

def test_rf_settings_main_property():
    rf, system, _ = _make_rf_stack()
    assert rf.main is system
