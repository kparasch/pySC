"""Tests for pySC.configuration.tuning_conf: configure_tuning."""
import pytest

from pySC.core.lattice import ATLattice
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.configuration.magnets_conf import configure_magnets
from pySC.configuration.tuning_conf import configure_tuning


def _make_sc_with_tuning_config(hmba_lattice_file):
    """Build a fresh SC with magnets and tuning configuration."""
    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")
    config = {
        "error_table": {
            "sext_cal": "0",
        },
        "magnets": {
            "sextupoles": {
                "regex": "^S[DF]",
                "components": [
                    {"B3": "sext_cal"},
                    {"B1": "sext_cal"},
                    {"A1": "sext_cal"},
                ],
            },
            "quadrupoles": {
                "regex": "^Q",
                "components": [{"B2": "sext_cal"}],
            },
        },
        "tuning": {
            "HCORR": [
                {"sextupoles": "B1"},
            ],
            "VCORR": [
                {"sextupoles": "A1"},
            ],
            "multipoles": [
                {"sextupoles": "B3"},
            ],
            "bba_magnets": [
                {"quadrupoles": "B2"},
            ],
            "tune": {
                "controls_1": {
                    "regex": "^QF1",
                    "component": "B2",
                },
                "controls_2": {
                    "regex": "^QD2",
                    "component": "B2",
                },
            },
        },
    }
    SC = SimulatedCommissioning(lattice=lattice, configuration=config, seed=42)
    return SC


# ---------------------------------------------------------------------------
# configure_tuning
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_configure_tuning(hmba_lattice_file):
    """configure_tuning populates HCORR, VCORR, multipoles, bba_magnets, and tune controls."""
    SC = _make_sc_with_tuning_config(hmba_lattice_file)
    configure_magnets(SC)
    configure_tuning(SC)

    # HCORR should contain sextupole B1 corrector controls
    assert len(SC.tuning.HCORR) > 0
    for name in SC.tuning.HCORR:
        assert "/B1" in name

    # VCORR should contain sextupole A1 corrector controls
    assert len(SC.tuning.VCORR) > 0
    for name in SC.tuning.VCORR:
        assert "/A1" in name

    # multipoles should contain sextupole B3 controls
    assert len(SC.tuning.multipoles) > 0
    for name in SC.tuning.multipoles:
        assert "/B3" in name

    # bba_magnets should contain quadrupole B2 controls
    assert len(SC.tuning.bba_magnets) > 0
    for name in SC.tuning.bba_magnets:
        assert "/B2" in name

    # tune controls should be populated
    assert len(SC.tuning.tune.controls_1) > 0
    assert len(SC.tuning.tune.controls_2) > 0
    for name in SC.tuning.tune.controls_1:
        assert "/B2" in name
    for name in SC.tuning.tune.controls_2:
        assert "/B2" in name
