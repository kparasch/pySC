"""Tests for pySC/apps/bba.py — beam-based alignment measurement and analysis."""

import pytest
import numpy as np

from pySC.apps.bba import (
    BBAData,
    BBA_Measurement,
    BBAAnalysis,
    hysteresis_loop,
    prep_ios,
    reject_bpm_outlier,
    reject_slopes,
    reject_center_outlier,
)
from pySC.apps.codes import BBACode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bba_data(n0=7, n_bpms=10, bpm_number=3, plane='H', bipolar=True,
                   dk0l=1e-4, dk1l=0.05, offset=0.0):
    """Build a BBAData with synthetic linear orbit data.

    Physics model:
    - Corrector steers beam across quad at positions k0[ii]
    - BPM at bpm_number reads: k0[ii] + offset (BPM has mechanical offset)
    - Quad strength change dk1l produces orbit kick ∝ beam_pos_at_quad = k0[ii]
    - Downstream BPMs see orbit shift ∝ response[j] * k0[ii]
    - Each BPM has a different response so centers have nonzero spread
    - BBA analysis extrapolates IOS→0 in BPM coordinates → finds offset
    """
    data = BBAData(
        quadrupole='Q1', bpm='BPM3', corrector='CH1', plane=plane,
        dk0l=dk0l, dk1l=dk1l, n0=n0, shots_per_orbit=1, bipolar=bipolar,
        bpm_number=bpm_number,
    )

    k0_array = np.linspace(-dk0l, dk0l, n0)  # corrector steps
    # Per-BPM response amplitudes (varied so centers have nonzero spread)
    rng = np.random.default_rng(123)
    response = 1.0 + 0.3 * rng.standard_normal(n_bpms)
    response = np.abs(response)  # keep positive

    for ii in range(n0):
        # BPM center orbit (no quad perturbation)
        x_center = np.zeros(n_bpms)
        # The measured BPM reads beam position + offset
        x_center[bpm_number] = k0_array[ii] + offset

        # IOS ∝ beam_pos_at_quad * response[j] — beam pos is k0[ii], not BPM reading
        ios = response * k0_array[ii]

        x_up = x_center.copy()
        x_up += ios  # orbit shift from +dk1l

        x_down = x_center.copy()
        if bipolar:
            x_down -= ios  # orbit shift from -dk1l

        y_center = np.zeros(n_bpms)
        y_up = np.zeros(n_bpms)
        y_down = np.zeros(n_bpms)

        data.raw_bpm_x_center.append(list(x_center))
        data.raw_bpm_y_center.append(list(y_center))
        data.raw_bpm_x_up.append(list(x_up))
        data.raw_bpm_y_up.append(list(y_up))
        data.raw_bpm_x_down.append(list(x_down))
        data.raw_bpm_y_down.append(list(y_down))

        data.raw_bpm_x_center_err.append(list(np.zeros(n_bpms)))
        data.raw_bpm_y_center_err.append(list(np.zeros(n_bpms)))
        data.raw_bpm_x_up_err.append(list(np.zeros(n_bpms)))
        data.raw_bpm_y_up_err.append(list(np.zeros(n_bpms)))
        data.raw_bpm_x_down_err.append(list(np.zeros(n_bpms)))
        data.raw_bpm_y_down_err.append(list(np.zeros(n_bpms)))

    return data


# ===========================================================================
# Pure analysis tests (no AT needed)
# ===========================================================================

class TestPrepIos:
    def test_prep_ios_known_linear_data(self):
        """prep_ios returns correct bpm_position and induced_orbit_shift for
        synthetic linear BPM data with a known offset."""
        offset = 0.002
        n0 = 7
        n_bpms = 10
        bpm_number = 3
        dk0l = 1e-4

        data = _make_bba_data(n0=n0, n_bpms=n_bpms, bpm_number=bpm_number,
                              offset=offset, bipolar=True)

        bpm_position, induced_orbit_shift = prep_ios(data)

        # bpm_position should be the mean across dk1l settings at the BPM
        # (center, up, down all include the bpm_number position).
        assert bpm_position.shape == (n0,)
        assert induced_orbit_shift.shape[0] == n0

        # bpm_position should vary linearly with the corrector steps
        expected_positions = np.linspace(-dk0l, dk0l, n0) + offset
        np.testing.assert_allclose(bpm_position, expected_positions, atol=1e-10)

    def test_prep_ios_bipolar_vs_unipolar(self):
        """Bipolar uses 3-point fit (down, center, up), unipolar uses 2-point
        fit (center, up)."""
        n0, n_bpms, bpm_number = 5, 8, 2
        data_bipolar = _make_bba_data(n0=n0, n_bpms=n_bpms,
                                       bpm_number=bpm_number, bipolar=True)
        data_unipolar = _make_bba_data(n0=n0, n_bpms=n_bpms,
                                        bpm_number=bpm_number, bipolar=False)

        _, ios_bi = prep_ios(data_bipolar)
        _, ios_uni = prep_ios(data_unipolar)

        # Both should produce finite results of the same shape
        assert ios_bi.shape == ios_uni.shape
        assert np.all(np.isfinite(ios_bi))
        assert np.all(np.isfinite(ios_uni))


class TestRejectBpmOutlier:
    def test_reject_bpm_outlier_no_outliers(self):
        """All BPMs within bounds gives an all-True mask."""
        rng = np.random.default_rng(42)
        ios = rng.normal(0, 1, (7, 10))
        mask = reject_bpm_outlier(ios, bpm_outlier_sigma=10.0)
        assert mask.shape == (10,)
        assert np.all(mask)

    def test_reject_bpm_outlier_one_outlier(self):
        """One BPM with a huge outlier is masked out."""
        # Use many samples so one outlier doesn't dominate mean/std
        rng = np.random.default_rng(42)
        ios = rng.normal(0, 1, (50, 10))
        # Inject a huge outlier at BPM index 5
        ios[25, 5] = 100.0
        mask = reject_bpm_outlier(ios, bpm_outlier_sigma=6.0)
        assert not mask[5], "BPM 5 should be rejected"
        # Others should still pass
        assert mask.sum() >= 9

    @pytest.mark.regression
    def test_reject_bpm_outlier_symmetric(self):
        """Both positive and negative outliers are rejected.

        Regression: the original code used ``data - mean > threshold``
        (no np.abs), so only positive outliers were caught.
        """
        rng = np.random.default_rng(99)
        ios = rng.normal(0, 1, (50, 5))
        # Inject a large NEGATIVE outlier at BPM 2
        ios[10, 2] = -100.0
        # Inject a large POSITIVE outlier at BPM 4
        ios[20, 4] = 100.0
        mask = reject_bpm_outlier(ios, bpm_outlier_sigma=6.0)
        assert not mask[2], "BPM 2 (negative outlier) should be rejected"
        assert not mask[4], "BPM 4 (positive outlier) should be rejected"


class TestRejectSlopes:
    def test_reject_slopes_cutoff(self):
        """Slopes below cutoff fraction of max are masked out."""
        slopes = np.array([1.0, 0.5, 0.05, 0.8, 0.01])
        mask = reject_slopes(slopes, slope_cutoff=0.1)
        # max |slope| = 1.0, cutoff = 0.1 * 1.0 = 0.1
        assert mask[0] and mask[1] and mask[3]  # above cutoff
        assert not mask[2] and not mask[4]  # below cutoff


class TestRejectCenterOutlier:
    def test_reject_center_outlier_cutoff(self):
        """Centers beyond cutoff sigma are masked out."""
        centers = np.array([0.0, 0.1, -0.1, 5.0, 0.05])
        mask = reject_center_outlier(centers, center_cutoff=1.0)
        # center at index 3 (5.0) is a clear outlier
        assert not mask[3], "Center at index 3 should be rejected"
        # Nearby centers should pass
        assert mask[0]


class TestBBAAnalysis:
    def test_bba_analysis_known_offset(self):
        """Synthetic data with known BPM offset d, BBAAnalysis.analyze()
        converges to d."""
        offset = 0.003
        data = _make_bba_data(n0=9, n_bpms=10, bpm_number=3,
                              offset=offset, bipolar=True)

        result = BBAAnalysis.analyze(data)
        np.testing.assert_allclose(result.offset, offset, atol=1e-6)

    def test_bba_analysis_error_propagation(self):
        """Error formula produces finite, positive error."""
        offset = 0.001
        data = _make_bba_data(n0=9, n_bpms=10, bpm_number=3,
                              offset=offset, bipolar=True)

        result = BBAAnalysis.analyze(data)
        assert np.isfinite(result.offset_error)
        assert result.offset_error >= 0

    @pytest.mark.regression
    def test_bba_analysis_total_rejections_stored(self):
        """BBAAnalysis.analyze() stores total_rejections correctly.

        Regression: the analyze() classmethod passes total_rejections to the
        constructor, but the field must actually exist on the model to be
        persisted.
        """
        data = _make_bba_data(n0=9, n_bpms=10, bpm_number=3,
                              offset=0.001, bipolar=True)

        result = BBAAnalysis.analyze(data)

        # The total_rejections attribute should be accessible and equal to
        # the count of False entries in mask_accepted.
        expected = int(np.sum(~result.mask_accepted))
        assert hasattr(result, 'total_rejections'), (
            "total_rejections is not stored on BBAAnalysis — "
            "field is missing from the model definition"
        )
        assert result.total_rejections == expected


# ===========================================================================
# Generator / measurement tests (need MockInterface)
# ===========================================================================

class TestHysteresisLoop:
    def test_hysteresis_loop_restores_setpoint(self, mock_interface):
        """After hysteresis loop completes, magnet is at original setpoint."""
        iface = mock_interface(n_bpms=5)
        iface.set('Q1', 1.5)

        codes = list(hysteresis_loop('Q1', iface, delta=0.1, n_cycles=2, bipolar=True))

        assert iface.get('Q1') == pytest.approx(1.5)

    def test_hysteresis_loop_yields_correct_codes(self, mock_interface):
        """Code sequence is HYSTERESIS, HYSTERESIS, ..., HYSTERESIS_DONE."""
        iface = mock_interface(n_bpms=5)
        iface.set('Q1', 0.0)

        codes = list(hysteresis_loop('Q1', iface, delta=0.1, n_cycles=2, bipolar=True))

        # n_cycles=2 bipolar => 2 HYSTERESIS per cycle (up, down) = 4 HYSTERESIS, then 1 HYSTERESIS_DONE
        assert all(c == BBACode.HYSTERESIS for c in codes[:-1])
        assert codes[-1] == BBACode.HYSTERESIS_DONE


class TestBBAMeasurement:
    def _make_measurement(self):
        return BBA_Measurement(
            bpm='BPM3',
            quadrupole='Q1',
            h_corrector='CH1',
            v_corrector='CV1',
            dk0l_x=1e-4,
            dk1l_x=0.05,
            dk0l_y=1e-4,
            dk1l_y=0.05,
            n0=3,
            bpm_number=3,
            shots_per_orbit=1,
            bipolar=True,
        )

    def test_bba_measurement_generate_h_plane(self, mock_interface):
        """Generator produces HYSTERESIS codes then HORIZONTAL codes then DONE."""
        iface = mock_interface(n_bpms=10)
        meas = self._make_measurement()

        codes = list(meas.generate(iface, plane='H'))

        assert BBACode.HYSTERESIS in codes
        assert BBACode.HYSTERESIS_DONE in codes
        assert BBACode.HORIZONTAL in codes
        assert BBACode.HORIZONTAL_DONE in codes
        assert codes[-1] == BBACode.DONE
        # No vertical codes
        assert BBACode.VERTICAL not in codes

    def test_bba_measurement_generate_both_planes(self, mock_interface):
        """Generator produces H then V plane codes."""
        iface = mock_interface(n_bpms=10)
        meas = self._make_measurement()

        codes = list(meas.generate(iface, plane=None))

        assert BBACode.HORIZONTAL in codes
        assert BBACode.HORIZONTAL_DONE in codes
        assert BBACode.VERTICAL in codes
        assert BBACode.VERTICAL_DONE in codes
        assert codes[-1] == BBACode.DONE

        # Horizontal should come before vertical
        h_done_idx = codes.index(BBACode.HORIZONTAL_DONE)
        v_first_idx = codes.index(BBACode.VERTICAL)
        assert h_done_idx < v_first_idx

    def test_bba_measurement_collects_data(self, mock_interface):
        """After running generator, H_data/V_data have populated raw arrays."""
        iface = mock_interface(n_bpms=10)
        meas = self._make_measurement()

        # Exhaust generator
        list(meas.generate(iface, plane=None))

        assert len(meas.H_data.raw_bpm_x_center) == meas.n0
        assert len(meas.H_data.raw_bpm_x_up) == meas.n0
        assert len(meas.H_data.raw_bpm_x_down) == meas.n0
        assert len(meas.V_data.raw_bpm_x_center) == meas.n0
        assert len(meas.V_data.raw_bpm_x_up) == meas.n0

    def test_bba_measurement_skip_cycle(self, mock_interface):
        """skip_cycle=True skips hysteresis codes."""
        iface = mock_interface(n_bpms=10)
        meas = self._make_measurement()

        codes = list(meas.generate(iface, plane='H', skip_cycle=True))

        assert BBACode.HYSTERESIS not in codes
        assert BBACode.HYSTERESIS_DONE not in codes
        assert BBACode.HORIZONTAL in codes
        assert codes[-1] == BBACode.DONE
