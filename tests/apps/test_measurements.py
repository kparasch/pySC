"""Tests for pySC/apps/measurements.py — public API: orbit_correction, measure_bba, measure_ORM, measure_dispersion."""

import pytest
import numpy as np

from pySC.apps.measurements import orbit_correction, measure_bba, measure_ORM, measure_dispersion
from pySC.apps.response_matrix import ResponseMatrix
from pySC.apps.codes import BBACode, ResponseCode, DispersionCode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rm(n_bpms=5, n_corr=4):
    """Build a small ResponseMatrix for orbit correction tests."""
    n_out = 2 * n_bpms  # H + V BPM readings
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((n_out, n_corr))
    input_names = [f"CH{i}" for i in range(n_corr // 2)] + [f"CV{i}" for i in range(n_corr // 2)]
    output_names = [f"BPM{i}_H" for i in range(n_bpms)] + [f"BPM{i}_V" for i in range(n_bpms)]
    input_planes = ["H"] * (n_corr // 2) + ["V"] * (n_corr // 2)
    output_planes = ["H"] * n_bpms + ["V"] * n_bpms
    return ResponseMatrix(
        matrix=matrix,
        input_names=input_names,
        output_names=output_names,
        input_planes=input_planes,
        output_planes=output_planes,
    )


def _bba_config():
    """Minimal BBA config dict for measure_bba."""
    return {
        'QUAD': 'Q1',
        'HCORR': 'CH0',
        'HCORR_delta': 1e-4,
        'QUAD_dk_H': 0.05,
        'VCORR': 'CV0',
        'VCORR_delta': 1e-4,
        'QUAD_dk_V': 0.05,
        'QUAD_is_skew': False,
        'number': 0,
    }


# ---------------------------------------------------------------------------
# orbit_correction tests
# ---------------------------------------------------------------------------

class TestOrbitCorrection:
    def test_computes_trims(self, mock_interface):
        """orbit_correction returns a dict of corrector trims."""
        iface = mock_interface(n_bpms=5)
        rm = _make_rm()
        trims = orbit_correction(iface, rm)
        assert isinstance(trims, dict)

    def test_apply_false(self, mock_interface):
        """apply=False does not modify setpoints."""
        iface = mock_interface(n_bpms=5)
        rm = _make_rm()
        # Record all setpoints
        sp_before = dict(iface._setpoints)
        orbit_correction(iface, rm, apply=False)
        assert iface._setpoints == sp_before

    def test_apply_true(self, mock_interface):
        """apply=True updates corrector setpoints."""
        iface = mock_interface(n_bpms=5)
        # Give it a non-zero orbit to correct
        iface._orbit_x = np.array([0.001, -0.002, 0.003, -0.001, 0.002])
        rm = _make_rm()
        orbit_correction(iface, rm, apply=True)
        # At least one corrector should have a setpoint now
        assert len(iface._setpoints) > 0

    def test_with_reference(self, mock_interface):
        """Reference orbit is subtracted before solving."""
        iface = mock_interface(n_bpms=5)
        iface._orbit_x = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        rm = _make_rm()
        ref = np.concatenate([iface._orbit_x, iface._orbit_y])
        # With reference == orbit, the residual is zero → trims ≈ 0
        trims = orbit_correction(iface, rm, reference=ref)
        for v in trims.values():
            assert abs(v) < 1e-12

    def test_with_rf(self, mock_interface):
        """rf=True returns n_inputs+1 trims including RF frequency."""
        iface = mock_interface(n_bpms=5)
        iface._orbit_x = np.ones(5) * 0.001
        rm = _make_rm()
        # Need rf_response set
        rm.rf_response = np.ones(rm._n_outputs) * 1e-3
        trims = orbit_correction(iface, rm, rf=True)
        assert 'rf' in trims


# ---------------------------------------------------------------------------
# measure_bba tests
# ---------------------------------------------------------------------------

class TestMeasureBBA:
    def test_yields_codes(self, mock_interface, tmp_path):
        """Generator yields (code, measurement) tuples."""
        iface = mock_interface(n_bpms=5)
        config = _bba_config()
        results = list(measure_bba(iface, 'BPM0', config, skip_save=True))
        assert len(results) > 0
        for code, meas in results:
            assert isinstance(code, BBACode)

    def test_config_validation(self, mock_interface):
        """Missing config keys raise AssertionError."""
        iface = mock_interface(n_bpms=5)
        bad_config = {'QUAD': 'Q1'}  # missing required keys
        with pytest.raises(AssertionError):
            list(measure_bba(iface, 'BPM0', bad_config, skip_save=True))


# ---------------------------------------------------------------------------
# measure_ORM tests
# ---------------------------------------------------------------------------

class TestMeasureORM:
    def test_yields_codes(self, mock_interface):
        """Generator yields ResponseCode tuples."""
        iface = mock_interface(n_bpms=5)
        correctors = ['CH0', 'CH1']
        results = list(measure_ORM(iface, correctors, delta=1e-4, skip_save=True))
        assert len(results) > 0
        for code, meas in results:
            assert isinstance(code, ResponseCode)


# ---------------------------------------------------------------------------
# measure_dispersion tests
# ---------------------------------------------------------------------------

class TestMeasureDispersion:
    def test_yields_codes(self, mock_interface):
        """Generator yields DispersionCode tuples."""
        iface = mock_interface(n_bpms=5)
        results = list(measure_dispersion(iface, delta=100.0, skip_save=True))
        assert len(results) > 0
        for code, meas in results:
            assert isinstance(code, DispersionCode)
