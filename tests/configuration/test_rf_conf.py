"""Tests for pySC.configuration.rf_conf: configure_rf."""
import pytest
import at

from pySC.core.lattice import ATLattice
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.configuration.rf_conf import configure_rf


def _make_sc_with_rf_config(hmba_lattice_file):
    """Build a fresh SC with an RF configuration that configure_rf can consume."""
    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")
    config = {
        "error_table": {},
        "rf": {
            "main": {
                "regex": "^RFC$",
            },
        },
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
def test_configure_rf_design_matches_error(hmba_lattice_file):
    """Both rf_settings and design_rf_settings are configured."""
    SC = _make_sc_with_rf_config(hmba_lattice_file)
    configure_rf(SC)

    # Both should have the same systems and cavities configured
    assert set(SC.rf_settings.systems.keys()) == set(SC.design_rf_settings.systems.keys())
    assert set(SC.rf_settings.cavities.keys()) == set(SC.design_rf_settings.cavities.keys())

    # The main system should have voltage/phase/frequency set
    main = SC.rf_settings.systems["main"]
    main_design = SC.design_rf_settings.systems["main"]
    assert main.voltage == pytest.approx(main_design.voltage)
    assert main.phase == pytest.approx(main_design.phase)
    assert main.frequency == pytest.approx(main_design.frequency)


@pytest.mark.regression
def test_configure_rf_frequency_key_spelling():
    """The frequency error lookup uses the correct key 'frequency', not 'frequncy'.

    Regression: rf_conf.py line 52 had a typo 'frequncy' instead of 'frequency',
    causing KeyError when the frequency error path was exercised.

    This test verifies the source code contains the correctly spelled key.
    """
    import inspect
    from pySC.configuration.rf_conf import configure_rf
    source = inspect.getsource(configure_rf)
    assert "frequncy" not in source, "Typo 'frequncy' still present in configure_rf"
    assert "rf_conf['frequency']" in source or 'rf_conf["frequency"]' in source, \
        "configure_rf should look up rf_conf['frequency']"
