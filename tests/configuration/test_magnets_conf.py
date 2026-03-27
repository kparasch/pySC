"""Tests for pySC.configuration.magnets_conf: configure_magnets."""
import pytest
import numpy as np
import at

from pySC.core.lattice import ATLattice
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.configuration.magnets_conf import configure_magnets


def _make_sc_with_magnet_config(hmba_lattice_file, cal_error_sigma="0.01"):
    """Build a fresh SC with a configuration dict that configure_magnets can consume."""
    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")
    config = {
        "error_table": {
            "quad_cal": cal_error_sigma,
            "sext_cal": cal_error_sigma,
        },
        "parameters": {
            "quad_limit": "100",
        },
        "magnets": {
            "quadrupoles": {
                "regex": "^Q",
                "components": [{"B2": "quad_cal"}],
                "limits": [{"B2": "quad_limit"}],
            },
            "sextupoles": {
                "regex": "^S[DF]",
                "components": [
                    {"B3": "sext_cal"},
                    {"B1": "sext_cal"},
                    {"A1": "sext_cal"},
                ],
            },
        },
    }
    SC = SimulatedCommissioning(lattice=lattice, configuration=config, seed=42)
    return SC


# ---------------------------------------------------------------------------
# configure_magnets
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_configure_magnets_creates_controls(hmba_lattice_file):
    """After configure_magnets, SC.magnet_settings.controls is populated."""
    SC = _make_sc_with_magnet_config(hmba_lattice_file)
    configure_magnets(SC)
    assert len(SC.magnet_settings.controls) > 0


@pytest.mark.slow
def test_configure_magnets_creates_magnet_arrays(hmba_lattice_file):
    """SC.magnet_arrays is populated with category -> name mapping."""
    SC = _make_sc_with_magnet_config(hmba_lattice_file)
    configure_magnets(SC)
    assert "quadrupoles" in SC.magnet_arrays
    assert "sextupoles" in SC.magnet_arrays
    assert len(SC.magnet_arrays["quadrupoles"]) > 0
    assert len(SC.magnet_arrays["sextupoles"]) > 0


@pytest.mark.slow
def test_configure_magnets_applies_calibration_errors(hmba_lattice_file):
    """Links have non-unity factors when the error table has non-zero calibration error."""
    SC = _make_sc_with_magnet_config(hmba_lattice_file, cal_error_sigma="0.05")
    configure_magnets(SC)

    # At least one link should have factor != 1
    non_unity = []
    for link_name, link in SC.magnet_settings.links.items():
        if link.error.factor != pytest.approx(1.0):
            non_unity.append(link_name)
            # Factor = 1 + normal(0, 0.05); should be within a reasonable range of 1.0
            assert 0.5 <= link.error.factor <= 1.5, \
                f"Link {link_name} factor={link.error.factor} implausibly far from 1.0 for sigma=0.05"
    assert len(non_unity) > 0, "Expected at least one link with factor != 1"


@pytest.mark.slow
def test_configure_magnets_design_has_unity_factor(hmba_lattice_file):
    """Design magnet settings links always have factor=1."""
    SC = _make_sc_with_magnet_config(hmba_lattice_file, cal_error_sigma="0.05")
    configure_magnets(SC)

    for link_name, link in SC.design_magnet_settings.links.items():
        assert link.error.factor == pytest.approx(1.0), (
            f"Design link {link_name} has factor={link.error.factor}, expected 1.0"
        )


@pytest.mark.slow
def test_configure_magnets_dipole_convention(hmba_lattice_file):
    """B1 on a dipole with BendingAngle uses the special offset convention."""
    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")

    # Use DQ1B which is a unique dipole name in the HMBA lattice (index 50)
    dip_name = "DQ1B"
    dip_index = 50
    # Verify it is indeed a dipole with bending angle
    assert lattice.is_dipole(dip_index), f"{dip_name} should be a dipole"

    config = {
        "error_table": {"dip_cal": "0.01"},
        "magnets": {
            "dipoles": {
                "regex": f"^{dip_name}$",
                "components": [{"B1": "dip_cal"}],
            },
        },
    }
    SC = SimulatedCommissioning(lattice=lattice, configuration=config, seed=42)
    configure_magnets(SC)

    # Check the link for this dipole has a non-zero offset (the special convention)
    dip_control = f"{dip_name}/B1"
    link_name = f"{dip_control}->{dip_control}"
    link = SC.magnet_settings.links[link_name]

    bending_angle = lattice.get_bending_angle(dip_index)
    magnet_length = lattice.get_length(dip_index)
    expected_offset = -bending_angle / magnet_length
    assert link.error.offset == pytest.approx(expected_offset, rel=1e-6)


@pytest.mark.slow
def test_configure_magnets_limits(hmba_lattice_file):
    """Controls have limits when configured."""
    SC = _make_sc_with_magnet_config(hmba_lattice_file)
    configure_magnets(SC)

    # Quadrupole controls should have limits
    for control_name in SC.control_arrays["quadrupoles"]:
        control = SC.magnet_settings.controls[control_name]
        assert control.limits is not None, f"{control_name} should have limits set"
        assert control.limits == (-100.0, 100.0)


@pytest.mark.slow
def test_configure_magnets_invert(hmba_lattice_file):
    """Component inversion applies negative factor."""
    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")
    config = {
        "error_table": {"sext_cal": "0"},
        "magnets": {
            "sextupoles": {
                "regex": "^SD1A$",
                "components": [
                    {"B3": "sext_cal"},
                    {"A1": "sext_cal"},
                ],
                "invert": ["A1"],
            },
        },
    }
    SC = SimulatedCommissioning(lattice=lattice, configuration=config, seed=42)
    configure_magnets(SC)

    # Find the A1 link for SD1A
    link_name = "SD1A/A1->SD1A/A1"
    link = SC.magnet_settings.links[link_name]
    # With zero calibration error, factor is normal_trunc(1, 0) = 1.0 then * -1 = -1.0
    assert link.error.factor == pytest.approx(-1.0)
