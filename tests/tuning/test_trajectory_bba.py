"""Tests for pySC.tuning.trajectory_bba: trajectory beam-based alignment."""
import pytest
import numpy as np

from pySC.tuning.trajectory_bba import (
    Trajectory_BBA_Configuration,
    trajectory_bba,
    get_mag_s_pos,
)


pytestmark = pytest.mark.slow


@pytest.fixture
def sc_with_traj_rm(sc_tuning):
    """SC with pre-calculated trajectory response matrix (needed for BBA config)."""
    sc = sc_tuning
    sc.tuning.calculate_model_trajectory_response_matrix(n_turns=1)
    return sc


def test_generate_trajectory_bba_config(sc_with_traj_rm):
    """Trajectory_BBA_Configuration.generate_config runs and produces config for each BPM."""
    sc = sc_with_traj_rm
    config = Trajectory_BBA_Configuration.generate_config(
        SC=sc,
        max_dx_at_bpm=1e-3,
        max_modulation=0.2e-3,
        n_downstream_bpms=50,
        max_ncorr_index=10,
    )
    assert isinstance(config, Trajectory_BBA_Configuration)
    # Should have one entry per BPM
    assert len(config.config) == len(sc.bpm_system.names)
    for bpm_name in sc.bpm_system.names:
        assert bpm_name in config.config
        entry = config.config[bpm_name]
        assert 'QUAD' in entry
        assert 'HCORR' in entry
        assert 'VCORR' in entry


@pytest.mark.xfail(reason="HMBA single-cell ring lacks sufficient BPMs downstream for trajectory BBA")
def test_trajectory_bba_single_bpm(sc_with_traj_rm):
    """trajectory_bba runs without error on a single BPM."""
    sc = sc_with_traj_rm
    bpm_name = sc.bpm_system.names[0]

    # Run trajectory BBA for horizontal plane
    offset, offset_err = trajectory_bba(
        sc, bpm_name, plane='H', shots_per_trajectory=1, n_corr_steps=3,
    )
    assert isinstance(offset, float)
    assert isinstance(offset_err, float)


def test_get_mag_s_pos(sc_with_traj_rm):
    """get_mag_s_pos returns s-positions for magnet controls."""
    sc = sc_with_traj_rm
    mag_controls = sc.tuning.bba_magnets[:3]
    s_positions = get_mag_s_pos(sc, mag_controls)
    assert len(s_positions) == 3
    # s-positions should be non-negative
    for s in s_positions:
        assert s >= 0
