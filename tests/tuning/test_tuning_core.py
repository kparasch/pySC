"""Tests for tune_scan() and synch_energy_correction() in tuning_core.py."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pySC.tuning.tuning_core import Tuning


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

        mock_ie = MagicMock(return_value=0.1)
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

        mock_ie = MagicMock(return_value=0.1)
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
                return 0.5
            return 0.1

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
                return 0.9
            return 0.1

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
            return 0.9  # Always above target

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

        mock_ie = MagicMock(return_value=0.0)
        with patch.object(Tuning, "injection_efficiency", mock_ie):
            _, _, _, error = tuning.tune_scan(
                dqx, dqy, n_turns=10, target=0.8, full_scan=True
            )
        assert error == 2
