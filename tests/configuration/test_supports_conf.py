"""Tests for pySC.configuration.supports_conf: configure_supports, generate_element_misalignments."""
import pytest
import numpy as np

from pySC.core.lattice import ATLattice
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.configuration.supports_conf import generate_element_misalignments, configure_supports
from pySC.configuration.magnets_conf import configure_magnets
from pySC.configuration.bpm_system_conf import configure_bpms


def _make_sc_with_support_config(hmba_lattice_file, dx_sigma="1e-4", dy_sigma="1e-4", roll_sigma="1e-5"):
    """Build a fresh SC with magnets, BPMs, and supports configuration."""
    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")
    config = {
        "error_table": {
            "quad_cal": "0",
            "mag_dx": dx_sigma,
            "mag_dy": dy_sigma,
            "mag_roll": roll_sigma,
            "support_dx": dx_sigma,
            "support_dy": dy_sigma,
            "support_roll": roll_sigma,
        },
        "magnets": {
            "quadrupoles": {
                "regex": "^Q",
                "components": [{"B2": "quad_cal"}],
                "dx": "mag_dx",
                "dy": "mag_dy",
                "roll": "mag_roll",
            },
        },
        "bpms": {
            "standard": {
                "regex": "^BPM",
            },
        },
        "supports": [
            {
                "level": 1,
                "start_endpoints": {"regex": "^QF1A$"},
                "end_endpoints": {"regex": "^QF1E$"},
                "dx": "support_dx",
                "dy": "support_dy",
                "roll": "support_roll",
            },
        ],
    }
    SC = SimulatedCommissioning(lattice=lattice, configuration=config, seed=42)
    return SC


# ---------------------------------------------------------------------------
# configure_supports
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_configure_supports_creates_levels(hmba_lattice_file):
    """Support system has L0 and L1 levels after full configuration."""
    SC = _make_sc_with_support_config(hmba_lattice_file)
    configure_magnets(SC)
    configure_bpms(SC)
    configure_supports(SC)

    assert "L0" in SC.support_system.data
    assert "L1" in SC.support_system.data
    # L0 should contain elements (from magnets and BPMs)
    assert len(SC.support_system.data["L0"]) > 0
    # L1 should contain at least the one support we configured
    assert len(SC.support_system.data["L1"]) > 0


@pytest.mark.slow
def test_configure_supports_applies_misalignments(hmba_lattice_file):
    """Support endpoints have non-zero offsets from the error table."""
    SC = _make_sc_with_support_config(hmba_lattice_file, dx_sigma="1e-4", dy_sigma="1e-4")
    configure_magnets(SC)
    configure_bpms(SC)
    configure_supports(SC)

    # Check L1 supports have been assigned dx/dy for both start and end endpoints
    has_nonzero_start = False
    has_nonzero_end = False
    for support_idx, support in SC.support_system.data["L1"].items():
        if support.start.dx != 0 or support.start.dy != 0:
            has_nonzero_start = True
        if support.end.dx != 0 or support.end.dy != 0:
            has_nonzero_end = True
    assert has_nonzero_start, "Expected at least one support with non-zero start endpoint misalignment"
    assert has_nonzero_end, "Expected at least one support with non-zero end endpoint misalignment"


# ---------------------------------------------------------------------------
# generate_element_misalignments
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_generate_element_misalignments(hmba_lattice_file):
    """Misalignment dx/dy/roll is applied to an element in the support system."""
    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")
    config = {
        "error_table": {
            "dx_err": "1e-4",
            "dy_err": "2e-4",
            "roll_err": "1e-5",
        },
    }
    SC = SimulatedCommissioning(lattice=lattice, configuration=config, seed=42)

    element_index = 5  # QF1A
    category_conf = {"dx": "dx_err", "dy": "dy_err", "roll": "roll_err"}
    generate_element_misalignments(SC, element_index, category_conf)

    elem = SC.support_system.data["L0"][element_index]
    # With non-zero sigmas and seed 42, at least some should be non-zero
    # The values are drawn from normal_trunc(0, sigma), so they should be small but non-zero
    assert elem.index == element_index
    # At least one of dx, dy, roll should be non-zero given non-zero sigmas
    assert elem.dx != 0 or elem.dy != 0 or elem.roll != 0
