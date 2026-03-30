"""Tests for pySC.configuration.rf_conf: configure_rf."""
import pytest
import at

from pySC.core.lattice import ATLattice
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.configuration.rf_conf import configure_rf


def _make_sc_with_rf_config(hmba_lattice_file, with_errors=False):
    """Build a fresh SC with an RF configuration that configure_rf can consume."""
    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")
    rf_category = {"regex": "^RFC$"}
    error_table = {}
    if with_errors:
        error_table = {
            "rf_voltage_err": "0.01",
            "rf_phase_err": "0.001",
            "rf_freq_err": "100",
        }
        rf_category["voltage"] = "rf_voltage_err"
        rf_category["phase"] = "rf_phase_err"
        rf_category["frequency"] = "rf_freq_err"
    config = {
        "error_table": error_table,
        "rf": {"main": rf_category},
    }
    SC = SimulatedCommissioning(lattice=lattice, configuration=config, seed=42)
    return SC


# ---------------------------------------------------------------------------
# configure_rf
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_configure_rf_creates_system(hmba_lattice_file):
    """RF system and cavities are created after configure_rf."""
    SC = _make_sc_with_rf_config(hmba_lattice_file)
    configure_rf(SC)

    assert "main" in SC.rf_settings.systems
    assert len(SC.rf_settings.cavities) >= 1
    assert "main" in SC.design_rf_settings.systems
    assert len(SC.design_rf_settings.cavities) >= 1


@pytest.mark.slow
def test_configure_rf_cavity_indices(hmba_lattice_file):
    """Cavity sim_index values match lattice RF cavity indices."""
    SC = _make_sc_with_rf_config(hmba_lattice_file)
    configure_rf(SC)

    # Get actual RF cavity indices from lattice
    ring = SC.lattice.design
    expected_rf_indices = [i for i, e in enumerate(ring) if isinstance(e, at.RFCavity)]

    # Get cavity indices from configured RF settings
    configured_indices = [cav.sim_index for cav in SC.rf_settings.cavities.values()]
    assert sorted(configured_indices) == sorted(expected_rf_indices)


@pytest.mark.slow
def test_configure_rf_design_and_operational_both_populated(hmba_lattice_file):
    """Both rf_settings and design_rf_settings are configured with matching structure."""
    SC = _make_sc_with_rf_config(hmba_lattice_file)
    configure_rf(SC)

    # Both should have the same systems and cavities configured
    assert set(SC.rf_settings.systems.keys()) == set(SC.design_rf_settings.systems.keys())
    assert set(SC.rf_settings.cavities.keys()) == set(SC.design_rf_settings.cavities.keys())

    # Without errors, the main system should have identical voltage/phase/frequency
    main = SC.rf_settings.systems["main"]
    main_design = SC.design_rf_settings.systems["main"]
    assert main.voltage == pytest.approx(main_design.voltage)
    assert main.phase == pytest.approx(main_design.phase)
    assert main.frequency == pytest.approx(main_design.frequency)


@pytest.mark.slow
@pytest.mark.regression
def test_configure_rf_applies_cavity_errors(hmba_lattice_file):
    """Cavity voltage/phase/frequency errors are applied from config.

    Regression: rf_conf.py previously looked up error keys on the top-level
    rf_conf dict (keyed by category names like 'main') instead of
    rf_category_conf (which contains 'voltage', 'phase', 'frequency' keys).
    This caused the error-application path to be silently skipped.
    """
    SC = _make_sc_with_rf_config(hmba_lattice_file, with_errors=True)
    configure_rf(SC)

    # At least one cavity should have non-zero errors applied
    has_voltage_error = False
    has_phase_error = False
    has_frequency_error = False
    for cav in SC.rf_settings.cavities.values():
        if cav.voltage_error != 0:
            has_voltage_error = True
        if cav.phase_error != 0:
            has_phase_error = True
        if cav.frequency_error != 0:
            has_frequency_error = True

    assert has_voltage_error, "Expected at least one cavity with non-zero voltage_error"
    assert has_phase_error, "Expected at least one cavity with non-zero phase_error"
    assert has_frequency_error, "Expected at least one cavity with non-zero frequency_error"


@pytest.mark.slow
@pytest.mark.regression
def test_configure_rf_design_differs_from_operational_with_errors(hmba_lattice_file):
    """Design cavities have zero errors while operational cavities have non-zero errors."""
    SC = _make_sc_with_rf_config(hmba_lattice_file, with_errors=True)
    configure_rf(SC)

    for name in SC.rf_settings.cavities:
        operational = SC.rf_settings.cavities[name]
        design = SC.design_rf_settings.cavities[name]
        # Design should always have zero errors
        assert design.voltage_error == 0
        assert design.phase_error == 0
        assert design.frequency_error == 0
        # Operational should have at least some non-zero error
        has_error = (operational.voltage_error != 0 or
                     operational.phase_error != 0 or
                     operational.frequency_error != 0)
        assert has_error, f"Cavity {name} has no errors applied"
