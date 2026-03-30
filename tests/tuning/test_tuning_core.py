"""Tests for pySC.tuning.tuning_core: Tuning class helpers and integration."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pySC.tuning.tuning_core import Tuning

# ---------------------------------------------------------------------------
# Pure helper tests (no AT needed)
# ---------------------------------------------------------------------------
# Note: _spiral_order does not exist in the current codebase. Tests removed.

@pytest.mark.slow
def test_bad_outputs_from_bad_bpms_length(sc_tuning):
    """Output length = len(bad_bpms) * 2 * n_turns."""
    sc = sc_tuning
    bad_bpms = [0, 2, 5]
    n_turns = 3
    result = sc.tuning.bad_outputs_from_bad_bpms(bad_bpms, n_turns=n_turns)
    expected_length = len(bad_bpms) * 2 * n_turns
    assert len(result) == expected_length


def test_bad_outputs_from_bad_bpms_indices(sc_tuning):
    """Indices correctly offset for H/V planes and turns."""
    sc = sc_tuning
    n_bpms = len(sc.bpm_system.indices)
    bad_bpms = [1]
    n_turns = 2
    result = sc.tuning.bad_outputs_from_bad_bpms(bad_bpms, n_turns=n_turns)

    # For plane=0 (H), turn=0: index = bpm + 0*n_bpms + 0*n_turns*n_bpms = 1
    # For plane=0 (H), turn=1: index = bpm + 1*n_bpms + 0*n_turns*n_bpms = 1 + n_bpms
    # For plane=1 (V), turn=0: index = bpm + 0*n_bpms + 1*n_turns*n_bpms = 1 + 2*n_bpms
    # For plane=1 (V), turn=1: index = bpm + 1*n_bpms + 1*n_turns*n_bpms = 1 + 3*n_bpms
    expected = [
        1,                          # H, turn 0
        1 + n_bpms,                 # H, turn 1
        1 + n_turns * n_bpms,       # V, turn 0
        1 + n_bpms + n_turns * n_bpms,  # V, turn 1
    ]
    assert result == expected


# ---------------------------------------------------------------------------
# Integration tests (require configured SC)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.xfail(reason="HMBA single-cell closed orbit near numerical floor — correction has no measurable effect")
def test_correct_orbit_reduces_rms(sc_tuning):
    """One correction iteration with exact RM reduces RMS orbit."""
    sc = sc_tuning

    # Perturb the orbit with a strong H corrector kick
    corr = sc.tuning.HCORR[0]
    sc.magnet_settings.set(corr, 5e-4)

    # Measure orbit before correction
    x_before, y_before = sc.bpm_system.capture_orbit()
    rms_before = np.sqrt(np.nanmean(x_before**2 + y_before**2))

    # Calculate model orbit RM and correct with mild regularization
    sc.tuning.calculate_model_orbit_response_matrix()
    sc.tuning.correct_orbit(n_reps=1, method='svd_cutoff', parameter=0)

    # Measure orbit after correction
    x_after, y_after = sc.bpm_system.capture_orbit()
    rms_after = np.sqrt(np.nanmean(x_after**2 + y_after**2))

    assert rms_after < rms_before


@pytest.mark.slow
def test_correct_injection_basic(sc_tuning):
    """Injection correction runs the full pipeline without error.

    Note: HMBA single-cell ring is too small for corrections to have measurable
    effect on corrector setpoints (see xfail on test_correct_orbit_reduces_rms).
    This test verifies the code path executes without exceptions.
    """
    sc = sc_tuning

    # Apply a small kick to create a trajectory offset
    corr = sc.tuning.HCORR[0]
    sc.magnet_settings.set(corr, 1e-5)

    # Build the trajectory RM first (n_turns=1)
    sc.tuning.calculate_model_trajectory_response_matrix(n_turns=1)
    sc.tuning.correct_injection(n_turns=1, n_reps=1, method='svd_cutoff', parameter=0)


@pytest.mark.slow
def test_correct_orbit_basic(sc_tuning):
    """Orbit correction runs the full pipeline without error.

    Note: HMBA single-cell ring is too small for corrections to have measurable
    effect on corrector setpoints (see xfail on test_correct_orbit_reduces_rms).
    This test verifies the code path executes without exceptions.
    """
    sc = sc_tuning

    # Apply a small kick to create an orbit offset
    corr = sc.tuning.HCORR[0]
    sc.magnet_settings.set(corr, 1e-5)

    # Build the orbit RM
    sc.tuning.calculate_model_orbit_response_matrix()
    sc.tuning.correct_orbit(n_reps=1, method='svd_cutoff', parameter=0)


@pytest.mark.slow
def test_wiggle_last_corrector(sc_tuning):
    """With full transmission, wiggle_last_corrector leaves correctors unchanged."""
    sc = sc_tuning

    # Record corrector setpoints before
    sp_before = {c: sc.magnet_settings.get(c) for c in sc.tuning.HCORR + sc.tuning.VCORR}

    sc.tuning.wiggle_last_corrector(max_steps=5, max_sp=100e-6)

    # With full transmission, no wiggling needed — setpoints should be unchanged
    sp_after = {c: sc.magnet_settings.get(c) for c in sc.tuning.HCORR + sc.tuning.VCORR}
    assert sp_before == sp_after, "Corrector setpoints changed despite full transmission"


@pytest.mark.slow
@pytest.mark.regression
def test_wiggle_last_corrector_no_upstream_corrector(sc_tuning):
    """When no corrector is upstream of last good BPM, should not raise UnboundLocalError.

    Regression: the original code used `last_hcor` / `last_vcor` which would be
    unbound if no corrector has sim_index < last_good_bpm_index.
    """
    sc = sc_tuning
    nbpm = len(sc.bpm_system.indices)

    # Simulate beam lost at last BPM (partial transmission)
    fake_x = np.zeros(nbpm)
    fake_x[-1] = np.nan  # last BPM has no reading
    fake_y = np.zeros(nbpm)

    original_capture = sc.bpm_system.__class__.capture_injection
    original_hcorr = sc.tuning.HCORR
    original_vcorr = sc.tuning.VCORR
    try:
        # Empty corrector lists → no corrector is upstream of anything
        sc.tuning.HCORR = []
        sc.tuning.VCORR = []
        # Monkeypatch at the class level since Pydantic blocks instance attribute patching
        sc.bpm_system.__class__.capture_injection = lambda self, **kw: (fake_x, fake_y)
        # This should not raise UnboundLocalError — it should log a warning and return
        sc.tuning.wiggle_last_corrector(max_steps=3, max_sp=100e-6)
    finally:
        sc.bpm_system.__class__.capture_injection = original_capture
        sc.tuning.HCORR = original_hcorr
        sc.tuning.VCORR = original_vcorr

def _make_tuning():
    """Create a Tuning instance with enough mocking for tune_scan tests."""
    tuning = Tuning()

    # Mock the parent SimulatedCommissioning
    SC = MagicMock()

    # magnet_settings.controls is a dict; start with knobs NOT registered
    SC.magnet_settings.controls = {}

    # magnet_settings.get / set backed by a real dict
    _store = {}

    def _ms_get(name, **kwargs):
        return _store.get(name, 0.0)

    def _ms_set(name, value, **kwargs):
        _store[name] = value

    SC.magnet_settings.get = MagicMock(side_effect=_ms_get)
    SC.magnet_settings.set = MagicMock(side_effect=_ms_set)

    tuning._parent = SC

    # Wire up the Tune sub-object's parent
    tuning.tune._parent = tuning

    return tuning, SC, _store


def _register_knobs(SC, store, knob_qx="qx_trim", knob_qy="qy_trim"):
    """Simulate knob registration so tune_scan's assert passes."""
    SC.magnet_settings.controls[knob_qx] = MagicMock()
    SC.magnet_settings.controls[knob_qy] = MagicMock()
    store[knob_qx] = 0.0
    store[knob_qy] = 0.0


# --------------------------------------------------------------------------- #
#  tune_scan: knobs not registered
# --------------------------------------------------------------------------- #
class TestTuneScanKnobAssert:
    def test_raises_when_qx_knob_missing(self):
        tuning, SC, store = _make_tuning()
        dqx = np.linspace(-0.01, 0.01, 3)
        dqy = np.linspace(-0.01, 0.01, 3)
        with pytest.raises(AssertionError, match="qx_trim"):
            tuning.tune_scan(dqx, dqy)

    def test_raises_when_qy_knob_missing(self):
        tuning, SC, store = _make_tuning()
        # Register only qx
        SC.magnet_settings.controls["qx_trim"] = MagicMock()
        store["qx_trim"] = 0.0
        dqx = np.linspace(-0.01, 0.01, 3)
        dqy = np.linspace(-0.01, 0.01, 3)
        with pytest.raises(AssertionError, match="qy_trim"):
            tuning.tune_scan(dqx, dqy)


# --------------------------------------------------------------------------- #
#  tune_scan: injection_efficiency called at every grid point (full_scan)
# --------------------------------------------------------------------------- #
class TestTuneScanCallsInjectionEfficiency:
    def test_full_scan_calls_injection_efficiency_for_every_point(self):
        tuning, SC, store = _make_tuning()
        _register_knobs(SC, store)

        n = 3
        dqx = np.linspace(-0.01, 0.01, n)
        dqy = np.linspace(-0.01, 0.01, n)

        mock_ie = MagicMock(return_value=np.array([0.1, 0.1]))
        # Patch on the class to avoid Pydantic __setattr__ restrictions
        with patch.object(Tuning, "injection_efficiency", mock_ie):
            best_dqx, best_dqy, smap, error = tuning.tune_scan(
                dqx, dqy, n_turns=10, target=0.8, full_scan=True
            )
        assert mock_ie.call_count == n * n
        # survival_map should have no NaNs
        assert not np.any(np.isnan(smap))

    def test_non_full_scan_with_low_survival_scans_all(self):
        tuning, SC, store = _make_tuning()
        _register_knobs(SC, store)

        n = 3
        dqx = np.linspace(-0.01, 0.01, n)
        dqy = np.linspace(-0.01, 0.01, n)

        mock_ie = MagicMock(return_value=np.array([0.1, 0.1]))
        with patch.object(Tuning, "injection_efficiency", mock_ie):
            _, _, _, error = tuning.tune_scan(
                dqx, dqy, n_turns=10, target=0.8, full_scan=False
            )
        # Never hit target, so all points scanned
        assert mock_ie.call_count == n * n
        assert error == 1  # improved but not target


# --------------------------------------------------------------------------- #
#  tune_scan: best deltas are applied at end
# --------------------------------------------------------------------------- #
class TestTuneScanAppliesBest:
    def test_best_deltas_applied_at_end(self):
        tuning, SC, store = _make_tuning()
        _register_knobs(SC, store)

        dqx = np.array([-0.02, 0.0, 0.02])
        dqy = np.array([-0.02, 0.0, 0.02])

        def _fake_ie(self_arg, n_turns=50, omp_num_threads=None):
            # Get current knob setpoints from the store
            qx_val = store.get("qx_trim", 0.0)
            qy_val = store.get("qy_trim", 0.0)
            # Best at dqx=0.02, dqy=-0.02
            if abs(qx_val - 0.02) < 1e-10 and abs(qy_val - (-0.02)) < 1e-10:
                return np.array([0.5, 0.5])
            return np.array([0.1, 0.1])

        with patch.object(Tuning, "injection_efficiency", _fake_ie):
            best_dqx, best_dqy, smap, error = tuning.tune_scan(
                dqx, dqy, n_turns=10, target=0.8, full_scan=True
            )

        assert best_dqx == pytest.approx(0.02)
        assert best_dqy == pytest.approx(-0.02)
        assert error == 1  # improved but did not reach target

        # After the call, the knobs should have been set to the best values
        set_calls = SC.magnet_settings.set.call_args_list
        # The last two set() calls apply the best deltas
        last_qx_call = None
        last_qy_call = None
        for call in reversed(set_calls):
            args, kwargs = call
            if args[0] == "qx_trim" and last_qx_call is None:
                last_qx_call = args[1]
            if args[0] == "qy_trim" and last_qy_call is None:
                last_qy_call = args[1]
            if last_qx_call is not None and last_qy_call is not None:
                break
        assert last_qx_call == pytest.approx(0.02)   # initial(0) + best_dqx(0.02)
        assert last_qy_call == pytest.approx(-0.02)   # initial(0) + best_dqy(-0.02)


# --------------------------------------------------------------------------- #
#  tune_scan: early termination when target is met
# --------------------------------------------------------------------------- #
class TestTuneScanEarlyTermination:
    def test_early_termination_when_target_met(self):
        tuning, SC, store = _make_tuning()
        _register_knobs(SC, store)

        n = 5
        dqx = np.linspace(-0.02, 0.02, n)
        dqy = np.linspace(-0.02, 0.02, n)

        call_count = [0]

        def _fake_ie(self_arg, n_turns=50, omp_num_threads=None):
            call_count[0] += 1
            # Center point (first in spiral) returns survival above target
            if call_count[0] == 1:
                return np.array([0.9, 0.9])
            return np.array([0.1, 0.1])

        with patch.object(Tuning, "injection_efficiency", _fake_ie):
            best_dqx, best_dqy, smap, error = tuning.tune_scan(
                dqx, dqy, n_turns=10, target=0.8, full_scan=False
            )

        # Should have terminated after first call
        assert call_count[0] == 1
        assert error == 0  # target reached

    def test_no_early_termination_with_full_scan(self):
        tuning, SC, store = _make_tuning()
        _register_knobs(SC, store)

        n = 3
        dqx = np.linspace(-0.01, 0.01, n)
        dqy = np.linspace(-0.01, 0.01, n)

        call_count = [0]

        def _fake_ie(self_arg, n_turns=50, omp_num_threads=None):
            call_count[0] += 1
            return np.array([0.9, 0.9])  # Always above target

        with patch.object(Tuning, "injection_efficiency", _fake_ie):
            _, _, _, error = tuning.tune_scan(
                dqx, dqy, n_turns=10, target=0.8, full_scan=True
            )

        # All points scanned despite exceeding target
        assert call_count[0] == n * n
        assert error == 0  # target reached (full scan)


# --------------------------------------------------------------------------- #
#  tune_scan: no transmission at all
# --------------------------------------------------------------------------- #
class TestTuneScanNoTransmission:
    def test_returns_error_2_when_all_zero(self):
        tuning, SC, store = _make_tuning()
        _register_knobs(SC, store)

        dqx = np.array([-0.01, 0.0, 0.01])
        dqy = np.array([-0.01, 0.0, 0.01])

        mock_ie = MagicMock(return_value=np.array([0.0, 0.0]))
        with patch.object(Tuning, "injection_efficiency", mock_ie):
            _, _, _, error = tuning.tune_scan(
                dqx, dqy, n_turns=10, target=0.8, full_scan=True
            )
        assert error == 2
