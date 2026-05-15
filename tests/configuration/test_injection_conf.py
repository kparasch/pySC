"""Tests for pySC.configuration.injection_conf: configure_injection."""
import pytest

from pySC.core.lattice import ATLattice
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.configuration.injection_conf import configure_injection


def _make_sc_for_injection(hmba_lattice_file, injection_conf=None):
    """Build a fresh SC with an optional injection configuration."""
    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")
    config = {}
    if injection_conf is not None:
        config["injection"] = injection_conf
    SC = SimulatedCommissioning(lattice=lattice, configuration=config, seed=42)
    return SC


# ---------------------------------------------------------------------------
# configure_injection
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_configure_injection(hmba_lattice_file):
    """configure_injection populates injection settings from design twiss."""
    SC = _make_sc_for_injection(hmba_lattice_file)
    configure_injection(SC)

    twiss = SC.lattice.twiss
    # Default is from_design=True, so injection params come from design twiss at s=0
    assert SC.injection.betx == pytest.approx(twiss["betx"][0])
    assert SC.injection.alfx == pytest.approx(twiss["alfx"][0])
    assert SC.injection.bety == pytest.approx(twiss["bety"][0])
    assert SC.injection.alfy == pytest.approx(twiss["alfy"][0])
    assert SC.injection.x == pytest.approx(twiss["x"][0])
    assert SC.injection.px == pytest.approx(twiss["px"][0])
    assert SC.injection.y == pytest.approx(twiss["y"][0])
    assert SC.injection.py == pytest.approx(twiss["py"][0])


@pytest.mark.slow
def test_configure_injection_with_overrides(hmba_lattice_file):
    """Injection config overrides design twiss and sets beam parameters."""
    injection_conf = {
        "from_design": True,
        "emit_x": "1e-9",
        "emit_y": "1e-11",
        "bunch_length": "3e-3",
        "energy_spread": "1e-3",
        "x_error_syst": "1e-4",
        "py_error_stat": "2e-5",
    }
    SC = _make_sc_for_injection(hmba_lattice_file, injection_conf=injection_conf)
    configure_injection(SC)

    assert SC.injection.gemit_x == pytest.approx(1e-9)
    assert SC.injection.gemit_y == pytest.approx(1e-11)
    assert SC.injection.bunch_length == pytest.approx(3e-3)
    assert SC.injection.energy_spread == pytest.approx(1e-3)
    assert SC.injection.x_error_syst == pytest.approx(1e-4)
    assert SC.injection.py_error_stat == pytest.approx(2e-5)


@pytest.mark.slow
def test_configure_injection_no_from_design(hmba_lattice_file):
    """With from_design=False, injection orbit params stay at defaults."""
    injection_conf = {
        "from_design": False,
    }
    SC = _make_sc_for_injection(hmba_lattice_file, injection_conf=injection_conf)

    # Record default betx before configure_injection
    default_betx = SC.injection.betx

    configure_injection(SC)

    # betx should remain at its default (1) because from_design is False
    assert SC.injection.betx == pytest.approx(default_betx)
