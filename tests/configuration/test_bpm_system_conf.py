"""Tests for pySC.configuration.bpm_system_conf: configure_bpms."""
import pytest
import numpy as np

from pySC.core.lattice import ATLattice
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.configuration.bpm_system_conf import configure_bpms


def _make_sc_with_bpm_config(hmba_lattice_file, cal_error="0.05", orbit_noise="1e-4", tbt_noise="1e-3"):
    """Build a fresh SC with a BPM configuration that configure_bpms can consume."""
    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")
    config = {
        "error_table": {
            "bpm_cal": cal_error,
            "orbit_noise": orbit_noise,
            "tbt_noise": tbt_noise,
        },
        "bpms": {
            "standard": {
                "regex": "^BPM",
                "calibration_error": "bpm_cal",
                "orbit_noise": "orbit_noise",
                "tbt_noise": "tbt_noise",
            },
        },
    }
    SC = SimulatedCommissioning(lattice=lattice, configuration=config, seed=42)
    return SC


# ---------------------------------------------------------------------------
# configure_bpms
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_configure_bpms_populates_indices(hmba_lattice_file):
    """bpm_system.indices is populated and sorted after configuration."""
    SC = _make_sc_with_bpm_config(hmba_lattice_file)
    configure_bpms(SC)

    assert len(SC.bpm_system.indices) == 10
    assert SC.bpm_system.indices == sorted(SC.bpm_system.indices)


@pytest.mark.slow
def test_configure_bpms_populates_names(hmba_lattice_file):
    """bpm_system.names matches lattice element names at BPM indices."""
    SC = _make_sc_with_bpm_config(hmba_lattice_file)
    configure_bpms(SC)

    for idx, name in zip(SC.bpm_system.indices, SC.bpm_system.names):
        expected = SC.lattice.get_name_from_index(idx)
        assert name == expected


@pytest.mark.slow
def test_configure_bpms_noise_arrays(hmba_lattice_file):
    """noise_co_x/y and noise_tbt_x/y have correct length and values."""
    SC = _make_sc_with_bpm_config(hmba_lattice_file, orbit_noise="1e-4", tbt_noise="1e-3")
    configure_bpms(SC)

    nbpm = len(SC.bpm_system.indices)
    assert len(SC.bpm_system.noise_co_x) == nbpm
    assert len(SC.bpm_system.noise_co_y) == nbpm
    assert len(SC.bpm_system.noise_tbt_x) == nbpm
    assert len(SC.bpm_system.noise_tbt_y) == nbpm

    np.testing.assert_array_almost_equal(SC.bpm_system.noise_co_x, 1e-4)
    np.testing.assert_array_almost_equal(SC.bpm_system.noise_co_y, 1e-4)
    np.testing.assert_array_almost_equal(SC.bpm_system.noise_tbt_x, 1e-3)
    np.testing.assert_array_almost_equal(SC.bpm_system.noise_tbt_y, 1e-3)


@pytest.mark.slow
def test_configure_bpms_calibration_errors(hmba_lattice_file):
    """Calibration errors are non-zero when configured with non-zero sigma."""
    SC = _make_sc_with_bpm_config(hmba_lattice_file, cal_error="0.05")
    configure_bpms(SC)

    nbpm = len(SC.bpm_system.indices)
    assert len(SC.bpm_system.calibration_errors_x) == nbpm
    assert len(SC.bpm_system.calibration_errors_y) == nbpm
    # With sigma=0.05 and 10 BPMs, it would be extremely unlikely for all to be exactly 0
    assert not np.allclose(SC.bpm_system.calibration_errors_x, 0)
    assert not np.allclose(SC.bpm_system.calibration_errors_y, 0)


@pytest.mark.slow
def test_configure_bpms_fields_initialized(hmba_lattice_file):
    """offsets_x/y, rolls, bba_offsets_x/y, reference_x/y are zeros of correct length."""
    SC = _make_sc_with_bpm_config(hmba_lattice_file)
    configure_bpms(SC)

    nbpm = len(SC.bpm_system.indices)
    for field in ['offsets_x', 'offsets_y', 'rolls', 'bba_offsets_x', 'bba_offsets_y',
                  'reference_x', 'reference_y']:
        arr = getattr(SC.bpm_system, field)
        assert len(arr) == nbpm, f"{field} has wrong length"
        np.testing.assert_array_equal(arr, np.zeros(nbpm), err_msg=f"{field} should be zeros")


@pytest.mark.slow
def test_configure_bpms_gain_corrections_ones(hmba_lattice_file):
    """gain_corrections_x/y are ones of correct length after configuration."""
    SC = _make_sc_with_bpm_config(hmba_lattice_file)
    configure_bpms(SC)

    nbpm = len(SC.bpm_system.indices)
    np.testing.assert_array_equal(SC.bpm_system.gain_corrections_x, np.ones(nbpm))
    np.testing.assert_array_equal(SC.bpm_system.gain_corrections_y, np.ones(nbpm))


@pytest.mark.slow
def test_configure_bpms_rot_matrices_updated(hmba_lattice_file):
    """_rot_matrices is not None after configuration."""
    SC = _make_sc_with_bpm_config(hmba_lattice_file)
    configure_bpms(SC)

    assert SC.bpm_system._rot_matrices is not None


@pytest.mark.slow
@pytest.mark.regression
def test_configure_bpms_multi_category_noise(hmba_lattice_file):
    """Two BPM categories with different noise values get per-category noise.

    Regression: nbpm was computed as len(bpms_indices) (cumulative count)
    instead of len(indices) (per-category count), so the second category's
    noise array was over-sized.
    """
    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")

    # Split BPMs into two categories with different noise values
    config = {
        "error_table": {
            "noise_a": "1e-4",
            "noise_b": "5e-4",
            "tbt_a": "1e-3",
            "tbt_b": "5e-3",
        },
        "bpms": {
            "cat_a": {
                "regex": "^BPM_0",  # BPMs whose name starts with BPM_0
                "orbit_noise": "noise_a",
                "tbt_noise": "tbt_a",
            },
            "cat_b": {
                "regex": "^BPM_1",  # BPMs whose name starts with BPM_1
                "orbit_noise": "noise_b",
                "tbt_noise": "tbt_b",
            },
        },
    }
    SC = SimulatedCommissioning(lattice=lattice, configuration=config, seed=42)
    configure_bpms(SC)

    total_bpms = len(SC.bpm_system.indices)
    # Arrays should have exactly the total number of BPMs (no over-count)
    assert len(SC.bpm_system.noise_co_x) == total_bpms
    assert len(SC.bpm_system.noise_tbt_x) == total_bpms

    # Each BPM should have its category's noise, not the cumulative count's noise
    for i, name in enumerate(SC.bpm_system.names):
        if name.startswith("BPM_0"):
            assert SC.bpm_system.noise_co_x[i] == pytest.approx(1e-4), \
                f"BPM {name} should have cat_a orbit noise"
            assert SC.bpm_system.noise_tbt_x[i] == pytest.approx(1e-3), \
                f"BPM {name} should have cat_a tbt noise"
        elif name.startswith("BPM_1"):
            assert SC.bpm_system.noise_co_x[i] == pytest.approx(5e-4), \
                f"BPM {name} should have cat_b orbit noise"
            assert SC.bpm_system.noise_tbt_x[i] == pytest.approx(5e-3), \
                f"BPM {name} should have cat_b tbt noise"
