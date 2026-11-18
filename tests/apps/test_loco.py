"""Tests for pySC.apps.loco — LOCO helpers and fitting."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pySC.apps.loco import (
    _objective,
    _get_parameters_mask,
    calculate_jacobian,
    loco_fit,
    apply_loco_corrections,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestGetParametersMask:
    """Test _get_parameters_mask boolean mask builder."""

    def test_all_flags_true(self):
        mask = _get_parameters_mask([True, True, True], [3, 4, 5])
        assert mask.shape == (12,)
        assert mask.all()

    def test_all_flags_false(self):
        mask = _get_parameters_mask([False, False, False], [3, 4, 5])
        assert not mask.any()

    def test_partial_flags(self):
        mask = _get_parameters_mask([True, False, True], [2, 3, 2])
        expected = np.array([True, True, False, False, False, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_single_group(self):
        mask = _get_parameters_mask([True], [5])
        assert mask.shape == (5,)
        assert mask.all()


class TestObjective:
    """Test _objective weighted residual computation."""

    def test_zero_params_returns_weighted_residual(self):
        """With zero parameter corrections, residual == orm_residuals * weights."""
        orm_res = np.ones((4, 3))
        jac = np.zeros((2, 4, 3))
        params = np.zeros(2)
        result = _objective(params, orm_res, jac, weights=1)
        np.testing.assert_allclose(result, orm_res.ravel())

    def test_perfect_correction_gives_zero(self):
        """If Jacobian * params == orm_residuals, residual is zero."""
        orm_res = np.ones((2, 2))
        # Single parameter, Jacobian entry is all ones
        jac = np.ones((1, 2, 2))
        params = np.array([1.0])
        result = _objective(params, orm_res, jac, weights=1)
        np.testing.assert_allclose(result, np.zeros(4), atol=1e-14)

    def test_weights_scale_residual(self):
        """Weights multiply the residual row-wise."""
        orm_res = np.ones((3, 2))
        jac = np.zeros((1, 3, 2))
        params = np.zeros(1)
        weights = np.array([1.0, 4.0, 9.0])
        result = _objective(params, orm_res, jac, weights)
        # sqrt(weights) applied row-wise: [1, 2, 3]
        expected = np.array([1, 1, 2, 2, 3, 3], dtype=float)
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Public API (with mocks for SC internals)
# ---------------------------------------------------------------------------

class TestCalculateJacobian:
    """Test calculate_jacobian structure and dimensions."""

    def test_corrector_and_bpm_blocks(self):
        """Corrector-gain and BPM-gain Jacobian blocks have correct structure."""
        SC = MagicMock()
        n_bpms, n_corr = 4, 3
        orm_model = np.random.RandomState(0).randn(n_bpms, n_corr)

        with patch("pySC.apps.loco.measure_OrbitResponseMatrix") as mock_orm:
            mock_orm.return_value = orm_model + 0.001  # small perturbation
            jac = calculate_jacobian(
                SC, orm_model, quad_control_names=["Q1"],
                include_correctors=True, include_bpms=True,
                include_dispersion=False, use_design=True,
            )

        # 1 quad + 3 correctors + 4 bpms = 8
        assert jac.shape == (1 + n_corr + n_bpms, n_bpms, n_corr)

    def test_no_correctors_no_bpms(self):
        """With correctors and BPMs disabled, only quad entries remain."""
        SC = MagicMock()
        orm_model = np.ones((2, 2))

        with patch("pySC.apps.loco.measure_OrbitResponseMatrix") as mock_orm:
            mock_orm.return_value = orm_model * 1.01
            jac = calculate_jacobian(
                SC, orm_model, quad_control_names=["Q1", "Q2"],
                include_correctors=False, include_bpms=False,
                include_dispersion=False,
            )

        assert jac.shape[0] == 2  # only quads


class TestLocoFit:
    """Test loco_fit returns correct structure."""

    def test_returns_all_correction_keys(self):
        SC = MagicMock()
        n_bpms, n_corr, n_quad = 4, 3, 2
        n_total = n_quad + n_corr + n_bpms
        orm_meas = np.random.RandomState(1).randn(n_bpms, n_corr)
        orm_model = orm_meas + 0.01
        jac = np.random.RandomState(2).randn(n_total, n_bpms, n_corr)

        result = loco_fit(SC, orm_meas, orm_model, jac)

        assert "quad_corrections" in result
        assert "corrector_corrections" in result
        assert "bpm_corrections" in result
        assert result["quad_corrections"].shape == (n_quad,)
        assert result["corrector_corrections"].shape == (n_corr,)
        assert result["bpm_corrections"].shape == (n_bpms,)

    def test_fit_subset(self):
        """Fitting only quads leaves corrector/bpm corrections at zero."""
        SC = MagicMock()
        n_bpms, n_corr, n_quad = 3, 2, 2
        n_total = n_quad + n_corr + n_bpms
        orm_meas = np.eye(n_bpms, n_corr) * 0.01
        orm_model = np.zeros((n_bpms, n_corr))
        jac = np.random.RandomState(3).randn(n_total, n_bpms, n_corr) * 0.1

        result = loco_fit(
            SC, orm_meas, orm_model, jac,
            fit_quads=True, fit_correctors=False, fit_bpms=False,
        )

        np.testing.assert_array_equal(result["corrector_corrections"], 0)
        np.testing.assert_array_equal(result["bpm_corrections"], 0)


class TestApplyLocoCorrections:
    """Test apply_loco_corrections calls magnet_settings correctly."""

    def test_applies_fraction(self):
        SC = MagicMock()
        SC.magnet_settings.get.return_value = 1.0

        corrections = {
            "quad_corrections": np.array([0.1, -0.2]),
            "corrector_corrections": np.array([]),
            "bpm_corrections": np.array([]),
        }

        apply_loco_corrections(SC, corrections, ["Q1", "Q2"], fraction=0.5)

        calls = SC.magnet_settings.set.call_args_list
        assert len(calls) == 2
        # Q1: 1.0 - 0.5*0.1 = 0.95
        np.testing.assert_allclose(calls[0].args[1], 0.95)
        # Q2: 1.0 - 0.5*(-0.2) = 1.1
        np.testing.assert_allclose(calls[1].args[1], 1.1)
