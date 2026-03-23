"""Tests for pySC.tuning.response_measurements: RM measurement wrappers."""
import pytest
import numpy as np

from pySC.tuning.response_measurements import (
    measure_TrajectoryResponseMatrix,
    measure_OrbitResponseMatrix,
)


pytestmark = pytest.mark.slow


def test_measure_trajectory_response_matrix(sc_tuning):
    """measure_TrajectoryResponseMatrix runs and returns correct shape."""
    sc = sc_tuning
    n_turns = 1
    n_bpms = len(sc.bpm_system.names)
    n_corr = len(sc.tuning.HCORR) + len(sc.tuning.VCORR)

    matrix = measure_TrajectoryResponseMatrix(sc, n_turns=n_turns, use_design=True)
    assert isinstance(matrix, np.ndarray)
    # Shape: (n_bpms * 2 * n_turns, n_corr)
    expected_rows = n_bpms * 2 * n_turns
    assert matrix.shape == (expected_rows, n_corr)
    assert np.all(np.isfinite(matrix))


def test_measure_orbit_response_matrix(sc_tuning):
    """measure_OrbitResponseMatrix runs and returns correct shape."""
    sc = sc_tuning
    n_bpms = len(sc.bpm_system.names)
    n_corr = len(sc.tuning.HCORR) + len(sc.tuning.VCORR)

    matrix = measure_OrbitResponseMatrix(sc, use_design=True)
    assert isinstance(matrix, np.ndarray)
    # Shape: (n_bpms * 2, n_corr)
    expected_rows = n_bpms * 2
    assert matrix.shape == (expected_rows, n_corr)
    assert np.all(np.isfinite(matrix))
