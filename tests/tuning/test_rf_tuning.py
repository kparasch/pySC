"""Tests for pySC.tuning.rf_tuning: RF frequency and phase optimization."""
import pytest
import numpy as np

from pySC.tuning.rf_tuning import RF_tuning


pytestmark = pytest.mark.slow


def test_measure_injection_phase(sc_tuning):
    """measure_injection_phase returns a finite array of tau values."""
    sc = sc_tuning
    n_turns = 3
    mean_tau = sc.tuning.rf.measure_injection_phase(n_turns=n_turns)
    assert isinstance(mean_tau, np.ndarray)
    # mean_tau has shape (n_particles, n_turns) — check n_turns axis
    assert mean_tau.shape[-1] == n_turns


def test_optimize_phase(sc_tuning):
    """optimize_phase runs and returns a phase value."""
    sc = sc_tuning
    best_phase = sc.tuning.rf.optimize_phase(
        low=-180, high=180, npoints=5, n_turns=3,
    )
    assert isinstance(best_phase, (float, np.floating))
    assert -180 <= best_phase <= 180
