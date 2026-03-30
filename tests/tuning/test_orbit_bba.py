"""Tests for pySC.tuning.orbit_bba: orbit beam-based alignment."""
import pytest
import numpy as np

from pySC.tuning.orbit_bba import (
    Orbit_BBA_Configuration,
    orbit_bba,
    get_mag_s_pos,
)


pytestmark = pytest.mark.slow


@pytest.fixture
def sc_with_orbit_rm(sc_tuning):
    """SC with pre-calculated orbit response matrix (needed for BBA config)."""
    sc = sc_tuning
    sc.tuning.calculate_model_orbit_response_matrix()
    return sc


@pytest.mark.xfail(reason="HMBA single-cell ring has near-zero response in some BPM/corrector pairs")
def test_generate_orbit_bba_config(sc_with_orbit_rm):
    """Orbit_BBA_Configuration.generate_config runs and produces config for each BPM."""
    sc = sc_with_orbit_rm
    config = Orbit_BBA_Configuration.generate_config(
        SC=sc,
        max_dx_at_bpm=0.3e-3,
        max_modulation=20e-6,
    )
    assert isinstance(config, Orbit_BBA_Configuration)
    # Should have one entry per BPM
    assert len(config.config) == len(sc.bpm_system.names)
    for bpm_name in sc.bpm_system.names:
        assert bpm_name in config.config
        entry = config.config[bpm_name]
        assert 'QUAD' in entry
        assert 'HCORR' in entry
        assert 'VCORR' in entry
        assert 'QUAD_dk_H' in entry
        assert 'QUAD_dk_V' in entry


@pytest.mark.xfail(reason="HMBA single-cell ring has near-zero response in some BPM/corrector pairs")
def test_orbit_bba_single_bpm(sc_with_orbit_rm):
    """orbit_bba runs without error on a single BPM."""
    sc = sc_with_orbit_rm
    bpm_name = sc.bpm_system.names[0]

    # Run orbit BBA for horizontal plane
    offset, offset_err = orbit_bba(
        sc, bpm_name, plane='H', shots_per_orbit=1, n_corr_steps=3,
    )
    assert isinstance(offset, float)
    assert isinstance(offset_err, float)


def test_orbit_bba_get_mag_s_pos(sc_with_orbit_rm):
    """get_mag_s_pos returns s-positions for magnet controls."""
    sc = sc_with_orbit_rm
    mag_controls = sc.tuning.HCORR[:2]
    s_positions = get_mag_s_pos(sc, mag_controls)
    assert len(s_positions) == 2
    for s in s_positions:
        assert isinstance(s, float)
        assert s >= 0
