"""Tests for pySC.apps.response_matrix — ResponseMatrix linear algebra core."""
import json
import numpy as np
import pytest
from pydantic import ValidationError

from pySC.apps.response_matrix import ResponseMatrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rm(n_out=6, n_in=4, seed=0, rf_response=None, rf_weight=None, randomize_rf=False, **kwargs):
    """Build a ResponseMatrix with a random matrix and sensible defaults."""
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n_out, n_in))
    input_names = kwargs.pop('input_names', [f'cor_{i}' for i in range(n_in)])
    output_names = kwargs.pop('output_names', [f'bpm_{i}' for i in range(n_out)])
    if rf_response is None and randomize_rf:
        rf_response = rng.normal(size=n_outputs)

    return ResponseMatrix(
        matrix=matrix,
        input_names=input_names,
        output_names=output_names,
        rf_response=rf_response,
        rf_weight=rf_weight,
        **kwargs,
    )

def _make_identity_rm(n=4, **kwargs):
    """Build a ResponseMatrix from an identity matrix (square, well-conditioned)."""
    matrix = np.eye(n)
    input_names = kwargs.pop('input_names', [f'cor_{i}' for i in range(n)])
    output_names = kwargs.pop('output_names', [f'bpm_{i}' for i in range(n)])
    input_planes = kwargs.pop('input_planes', ['H'] * (n // 2) + ['V'] * (n - n // 2))
    output_planes = kwargs.pop('output_planes', ['H'] * (n // 2) + ['V'] * (n - n // 2))
    return ResponseMatrix(
        matrix=matrix,
        input_names=input_names,
        output_names=output_names,
        input_planes=input_planes,
        output_planes=output_planes,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_construction_sets_dimensions(self):
        rm = _make_rm(n_out=8, n_in=6)
        assert rm._n_outputs == 8
        assert rm._n_inputs == 6

    def test_singular_values_computed(self):
        rm = _make_rm()
        assert rm.singular_values is not None
        assert len(rm.singular_values) == min(rm._n_outputs, rm._n_inputs)

    def test_default_plane_assignment(self):
        """With even inputs/outputs and no planes specified, defaults to H+V split."""
        rm = _make_rm(n_out=6, n_in=4)
        assert rm.input_planes == ['H', 'H', 'V', 'V']
        assert rm.output_planes == ['H', 'H', 'H', 'V', 'V', 'V']

    def test_default_weights_are_ones(self):
        rm = _make_rm(n_out=6, n_in=4)
        np.testing.assert_array_equal(rm.input_weights, np.ones(4))
        np.testing.assert_array_equal(rm.output_weights, np.ones(6))

    def test_default_rf_response_zeros(self):
        rm = _make_rm(n_out=6, n_in=4)
        np.testing.assert_array_equal(rm.rf_response, np.zeros(6))


# ---------------------------------------------------------------------------
# Bad inputs/outputs and caching
# ---------------------------------------------------------------------------

class TestBadInputsOutputs:
    def test_bad_inputs_setter_invalidates_cache(self):
        rm = _make_identity_rm(n=4)
        output = np.array([1.0, 2.0, 3.0, 4.0])
        # Trigger a solve to populate the cache
        rm.solve(output)
        assert rm._inverse_RM is not None

        # Setting bad_inputs should clear the cache
        rm.bad_inputs = [0]
        assert rm._inverse_RM is None

    def test_bad_outputs_setter_invalidates_cache(self):
        rm = _make_identity_rm(n=4)
        output = np.array([1.0, 2.0, 3.0, 4.0])
        rm.solve(output)
        assert rm._inverse_RM is not None

        rm.bad_outputs = [0]
        assert rm._inverse_RM is None


# ---------------------------------------------------------------------------
# Enable / disable by name
# ---------------------------------------------------------------------------

class TestEnableDisable:
    def test_disable_enable_inputs_by_name(self):
        rm = _make_rm(n_out=6, n_in=4)
        assert rm.bad_inputs == []

        rm.disable_inputs(['cor_1'])
        assert 1 in rm.bad_inputs

        rm.enable_inputs(['cor_1'])
        assert rm.bad_inputs == []

    def test_disable_enable_outputs_by_name(self):
        rm = _make_rm(n_out=6, n_in=4)
        assert rm.bad_outputs == []

        rm.disable_outputs(['bpm_2'])
        assert 2 in rm.bad_outputs

        rm.enable_outputs(['bpm_2'])
        assert rm.bad_outputs == []

    def test_disable_all_inputs_but(self):
        rm = _make_rm(n_out=6, n_in=4)
        rm.disable_all_inputs_but(['cor_0'])
        # Only cor_0 should remain enabled
        assert 0 not in rm.bad_inputs
        assert set(rm.bad_inputs) == {1, 2, 3}

    @pytest.mark.regression
    def test_disable_all_outputs_but(self):
        """Regression: disable_all_outputs_but() previously operated on inputs
        instead of outputs (copy-paste bug). Now correctly disables all outputs
        except the specified ones.
        """
        rm = _make_rm(n_out=6, n_in=4)

        rm.disable_all_outputs_but(['bpm_0'])

        # Only bpm_0 should remain enabled; all others disabled
        assert 0 not in rm.bad_outputs
        assert set(rm.bad_outputs) == {1, 2, 3, 4, 5}
        # Inputs should be untouched
        assert rm.bad_inputs == []


# ---------------------------------------------------------------------------
# SVD-based solve methods
# ---------------------------------------------------------------------------

class TestSolveSVD:
    def test_svd_cutoff_identity(self):
        """solve with identity RM and svd_cutoff(parameter=0) returns +output."""
        n = 4
        rm = _make_identity_rm(n=n)
        output = np.array([1.0, -2.0, 3.0, -4.0])
        result = rm.solve(output, method='svd_cutoff', parameter=0)
        # With identity RM and unit weights, solve returns +output
        np.testing.assert_array_almost_equal(result, output)

    def test_svd_cutoff_rank_reduction(self):
        """With parameter=0.5, only SVs above 50% of max are kept."""
        n = 4
        # Build a diagonal matrix with known singular values
        s = np.array([10.0, 6.0, 3.0, 1.0])  # 50% of max=10 is 5
        matrix = np.diag(s)
        rm = ResponseMatrix(
            matrix=matrix,
            input_names=[f'cor_{i}' for i in range(n)],
            output_names=[f'bpm_{i}' for i in range(n)],
            input_planes=['H', 'H', 'V', 'V'],
            output_planes=['H', 'H', 'V', 'V'],
        )
        output = np.ones(n)

        # With cutoff=0.5, keep SVs > 0.5 * 10 = 5, so keep s[0]=10 and s[1]=6
        result_cut = rm.solve(output, method='svd_cutoff', parameter=0.5)
        # With cutoff=0 (keep all)
        result_full = rm.solve(output, method='svd_cutoff', parameter=0)

        # The cut result should differ from full (fewer SVs used)
        assert not np.allclose(result_cut, result_full)

        # For a diagonal RM, the pseudoinverse is diag(1/s_i) for kept SVs.
        # solve() returns pseudoinverse @ output * input_weights (weights=1).
        # Kept SVs (0, 1): result[i] = 1 / s[i]
        # Dropped SVs (2, 3): result[i] = 0
        np.testing.assert_almost_equal(result_cut[0], 1.0 / 10.0)
        np.testing.assert_almost_equal(result_cut[1], 1.0 / 6.0)
        np.testing.assert_almost_equal(result_cut[2], 0.0)
        np.testing.assert_almost_equal(result_cut[3], 0.0)

    def test_svd_values_keeps_n(self):
        """method='svd_values', parameter=3 keeps exactly 3 singular values."""
        n = 6
        rm = _make_rm(n_out=n, n_in=n)

        # Build pseudoinverse directly to inspect
        inv_rm = rm.build_pseudoinverse(method='svd_values', parameter=3)
        # The inverse matrix shape should use 3 singular values
        # We can verify by checking the effective rank of the inverse
        # The inverse is V_k @ diag(1/s_k) @ U_k^T with k=3
        U, s, Vh = np.linalg.svd(inv_rm.matrix, full_matrices=False)
        # Only 3 non-negligible singular values
        n_nonzero = np.sum(s > 1e-12)
        assert n_nonzero == 3


# ---------------------------------------------------------------------------
# Tikhonov regularization
# ---------------------------------------------------------------------------

class TestTikhonov:
    def test_tikhonov_regularization(self):
        """Small alpha approximates pinv; large alpha shrinks correction."""
        n = 4
        rm = _make_identity_rm(n=n)
        output = np.ones(n)

        result_small = rm.solve(output, method='tikhonov', parameter=1e-10)
        result_large = rm.solve(output, method='tikhonov', parameter=1e3)

        # Small alpha ~ pseudo-inverse
        np.testing.assert_array_almost_equal(result_small, output, decimal=5)
        # Large alpha produces much smaller correction
        assert np.linalg.norm(result_large) < np.linalg.norm(result_small) * 0.01

    def test_tikhonov_alpha_zero_equals_svd(self):
        """Tikhonov with alpha=0 produces same result as svd_cutoff with parameter=0."""
        rm = _make_rm(n_out=6, n_in=4, seed=7)
        output = np.random.default_rng(99).standard_normal(6)

        result_tikh = rm.solve(output, method='tikhonov', parameter=0)
        result_svd = rm.solve(output, method='svd_cutoff', parameter=0)

        np.testing.assert_array_almost_equal(result_tikh, result_svd)


# ---------------------------------------------------------------------------
# Moore-Penrose property
# ---------------------------------------------------------------------------

class TestMoorePenrose:
    def test_moore_penrose_property(self):
        """For RM R, R @ R_inv @ R ~ R (pseudoinverse property)."""
        rm = _make_rm(n_out=6, n_in=4, seed=3)
        R = rm.matrix
        inv_rm = rm.build_pseudoinverse(method='svd_cutoff', parameter=0)
        R_inv = inv_rm.matrix

        # R @ R_inv @ R should approximately equal R
        np.testing.assert_array_almost_equal(R @ R_inv @ R, R, decimal=10)


# ---------------------------------------------------------------------------
# MICADO
# ---------------------------------------------------------------------------

class TestMICADO:
    def test_micado_single_corrector(self):
        """MICADO with n=1 selects the best single corrector."""
        n_out, n_in = 6, 4
        rm = _make_rm(n_out=n_out, n_in=n_in, seed=5)
        output = np.random.default_rng(10).standard_normal(n_out)

        result = rm.solve(output, method='micado', parameter=1)
        # Exactly one corrector should be non-zero
        assert np.sum(np.abs(result) > 1e-15) == 1

    def test_micado_full_correction(self):
        """MICADO with n=n_inputs gives similar result to full pinv for well-conditioned RM."""
        n = 4
        rm = _make_identity_rm(n=n)
        output = np.array([1.0, 2.0, 3.0, 4.0])

        result_micado = rm.solve(output, method='micado', parameter=n)
        result_pinv = rm.solve(output, method='svd_cutoff', parameter=0)

        np.testing.assert_array_almost_equal(result_micado, result_pinv, decimal=10)

    @pytest.mark.regression
    def test_micado_all_plane_inputs_disabled(self):
        """Regression: when all H-plane inputs are disabled, MICADO previously
        raised UnboundLocalError. Now returns empty correction dict.
        """
        n_out, n_in = 4, 4
        rm = _make_identity_rm(n=n_out)
        output = np.ones(n_out)

        # Disable all H-plane inputs (indices 0, 1 for a 4x4 matrix with
        # default plane split ['H', 'H', 'V', 'V'])
        h_indices = [i for i, p in enumerate(rm.input_planes) if p == 'H']
        rm.bad_inputs = h_indices

        # Should not raise; returns zero correction since no inputs available
        result = rm.solve(output, method='micado', parameter=1, plane='H')
        np.testing.assert_array_equal(result, 0.0)


# ---------------------------------------------------------------------------
# Solve caching
# ---------------------------------------------------------------------------

class TestSolveCaching:
    def test_solve_caching(self):
        """Calling solve() twice with same params reuses cached inverse."""
        rm = _make_identity_rm(n=4)
        output = np.ones(4)

        rm.solve(output, method='svd_cutoff', parameter=0)
        cached = rm._inverse_RM
        assert cached is not None

        rm.solve(output, method='svd_cutoff', parameter=0)
        # Same object should be reused (not rebuilt)
        assert rm._inverse_RM is cached

    def test_solve_cache_invalidation(self):
        """Changing weights between calls forces rebuild of inverse."""
        rm = _make_identity_rm(n=4)
        output = np.ones(4)

        rm.solve(output, method='svd_cutoff', parameter=0)
        cached = rm._inverse_RM
        assert cached is not None

        # Change a weight
        rm.input_weights[0] = 2.0
        rm.solve(output, method='svd_cutoff', parameter=0)
        # Cache should have been rebuilt (different hash)
        assert rm._inverse_RM is not cached


# ---------------------------------------------------------------------------
# Plane-specific solve
# ---------------------------------------------------------------------------

class TestSolvePlane:
    def test_solve_plane_h_only(self):
        """solve(plane='H') returns zeros in V-plane entries."""
        n = 6
        rm = _make_rm(n_out=n, n_in=n, seed=11)
        output = np.ones(n)

        result = rm.solve(output, method='svd_cutoff', parameter=0, plane='H')

        # V-plane input indices should be zero
        v_indices = [i for i, p in enumerate(rm.input_planes) if p == 'V']
        for idx in v_indices:
            assert result[idx] == 0.0

        # H-plane input indices should generally be non-zero
        h_indices = [i for i, p in enumerate(rm.input_planes) if p == 'H']
        h_values = result[h_indices]
        assert np.any(np.abs(h_values) > 1e-15)


# ---------------------------------------------------------------------------
# Virtual constraint
# ---------------------------------------------------------------------------

class TestVirtualConstraint:
    def test_solve_virtual_constraint(self):
        """solve(virtual=True) pushes the sum of corrector kicks toward zero.

        The virtual constraint adds a row to the response matrix that penalises
        non-zero sum.  With default virtual_weight=1 and a random RM the sum
        won't be exactly zero, but it should be substantially smaller than
        without the constraint.
        """
        n = 6
        rm = _make_rm(n_out=n, n_in=n, seed=22)
        output = np.ones(n)

        result_virtual = rm.solve(output, method='svd_cutoff', parameter=0, virtual=True)
        result_normal = rm.solve(output, method='svd_cutoff', parameter=0, virtual=False)

        # With virtual constraint, the sum should be significantly smaller
        sum_virtual = abs(np.sum(result_virtual))
        sum_normal = abs(np.sum(result_normal))
        assert sum_virtual < sum_normal


# ---------------------------------------------------------------------------
# RF column
# ---------------------------------------------------------------------------

class TestRFSolve:
    def test_solve_rf_column(self):
        """solve(rf=True) returns n_inputs+1 length array with RF correction."""
        n_out, n_in = 6, 4
        rng = np.random.default_rng(33)
        matrix = rng.standard_normal((n_out, n_in))
        rf_response = rng.standard_normal(n_out)

        rm = ResponseMatrix(
            matrix=matrix,
            input_names=[f'cor_{i}' for i in range(n_in)],
            output_names=[f'bpm_{i}' for i in range(n_out)],
            rf_response=rf_response,
        )
        output = rng.standard_normal(n_out)

        result = rm.solve(output, method='svd_cutoff', parameter=0, rf=True)
        assert len(result) == n_in + 1
        # The last element is the RF correction (scaled by rf_weight)
        # It should generally be non-zero for a random rf_response
        assert result[-1] != 0.0


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

class TestWeights:
    def test_set_weight_applies(self):
        """set_weight(name, weight) modifies the correct weight entry."""
        rm = _make_rm(n_out=6, n_in=4)
        rm.set_weight('cor_2', 5.0)
        assert rm.input_weights[2] == 5.0
        # Other weights unchanged
        assert rm.input_weights[0] == 1.0
        assert rm.input_weights[1] == 1.0
        assert rm.input_weights[3] == 1.0

    def test_set_weight_applies_to_output(self):
        """set_weight also works for output names."""
        rm = _make_rm(n_out=6, n_in=4)
        rm.set_weight('bpm_3', 0.5)
        assert rm.output_weights[3] == 0.5

    def test_default_rf_weight_calculation(self):
        """default_rf_weight = mean(std per H-plane input col of matrix_h) / std(rf_response).

        Uses H-plane submatrix only, not the full matrix.
        """
        n_out, n_in = 6, 4
        rng = np.random.default_rng(44)
        matrix = rng.standard_normal((n_out, n_in))
        rf_response = rng.standard_normal(n_out)

        rm = ResponseMatrix(
            matrix=matrix,
            input_names=[f'cor_{i}' for i in range(n_in)],
            output_names=[f'bpm_{i}' for i in range(n_out)],
            rf_response=rf_response,
        )

        # Compute expected default_rf_weight manually
        # H-plane outputs: first 3 (n_out//2), H-plane inputs: first 2 (n_in//2)
        matrix_h = matrix[:3, :2]  # H outputs x H inputs
        rms_per_input = np.std(matrix_h, axis=0)
        mean_rms = np.mean(rms_per_input)
        rms_rf = np.std(rf_response)
        expected = mean_rms / rms_rf

        np.testing.assert_almost_equal(rm.rf_weight, expected)


# ---------------------------------------------------------------------------
# JSON roundtrip
# ---------------------------------------------------------------------------

class TestJSON:
    def test_json_roundtrip(self, tmp_path):
        """to_json then from_json preserves matrix and metadata."""
        n_out, n_in = 6, 4
        rng = np.random.default_rng(55)
        matrix = rng.standard_normal((n_out, n_in))
        rf_response = rng.standard_normal(n_out)

        rm = ResponseMatrix(
            matrix=matrix,
            input_names=[f'cor_{i}' for i in range(n_in)],
            output_names=[f'bpm_{i}' for i in range(n_out)],
            input_planes=['H', 'H', 'V', 'V'],
            output_planes=['H', 'H', 'H', 'V', 'V', 'V'],
            rf_response=rf_response,
        )

        filepath = str(tmp_path / 'test_rm.json')
        rm.to_json(filepath)
        rm2 = ResponseMatrix.from_json(filepath)

        np.testing.assert_array_almost_equal(rm2.matrix, rm.matrix)
        np.testing.assert_array_almost_equal(rm2.rf_response, rm.rf_response)
        assert rm2.input_names == rm.input_names
        assert rm2.output_names == rm.output_names
        assert rm2.input_planes == rm.input_planes
        assert rm2.output_planes == rm.output_planes
        np.testing.assert_array_almost_equal(rm2.input_weights, rm.input_weights)
        np.testing.assert_array_almost_equal(rm2.output_weights, rm.output_weights)


# ---------------------------------------------------------------------------
# Deprecation handling
# ---------------------------------------------------------------------------

class TestDeprecation:
    def test_deprecation_inputs_plane_rename(self):
        """Old 'inputs_plane' key gets renamed to 'input_planes'."""
        n_out, n_in = 4, 4
        rm = ResponseMatrix(
            matrix=np.eye(n_out),
            input_names=[f'cor_{i}' for i in range(n_in)],
            output_names=[f'bpm_{i}' for i in range(n_out)],
            inputs_plane=['H', 'H', 'V', 'V'],
            output_planes=['H', 'H', 'V', 'V'],
        )
        assert rm.input_planes == ['H', 'H', 'V', 'V']

    def test_deprecation_outputs_plane_rename(self):
        """Old 'outputs_plane' key gets renamed to 'output_planes'."""
        n_out, n_in = 4, 4
        rm = ResponseMatrix(
            matrix=np.eye(n_out),
            input_names=[f'cor_{i}' for i in range(n_in)],
            output_names=[f'bpm_{i}' for i in range(n_out)],
            input_planes=['H', 'H', 'V', 'V'],
            outputs_plane=['H', 'H', 'V', 'V'],
        )
        assert rm.output_planes == ['H', 'H', 'V', 'V']


# ---------------------------------------------------------------------------
# Extra forbid (Pydantic strict mode)
# ---------------------------------------------------------------------------

class TestExtraForbid:
    def test_extra_forbid_rejects_unknown_kwargs(self):
        """Unknown kwargs raise ValidationError when extra='forbid' is set."""
        with pytest.raises(ValidationError):
            ResponseMatrix(
                matrix=np.eye(4),
                input_names=[f'cor_{i}' for i in range(4)],
                output_names=[f'bpm_{i}' for i in range(4)],
                nonexistent_field=42,
            )

# ---------------------------------------------------------------------------
# Test rf_response
# ---------------------------------------------------------------------------

class TestSetRfResponseRecomputesWeight:
    """set_rf_response() recomputes rf_weight when it was auto-computed."""

    def test_auto_weight_is_recomputed(self):
        rm = _make_rm(randomize_rf=True)
        assert rm._rf_weight_is_default is True
        old_weight = rm.rf_weight

        new_rf = np.random.default_rng(99).normal(size=rm.matrix.shape[0]) * 5
        rm.set_rf_response(new_rf)

        assert rm.rf_weight != old_weight
        expected = rm.default_rf_weight()
        assert rm.rf_weight == pytest.approx(expected)


class TestSetRfResponsePreservesExplicitWeight:
    """set_rf_response() does NOT overwrite an explicitly-set rf_weight."""

    def test_explicit_weight_preserved(self):
        explicit_weight = 5.0
        rm = _make_rm(rf_weight=explicit_weight, randomize_rf=True)
        assert rm._rf_weight_is_default is False
        assert rm.rf_weight == explicit_weight

        new_rf = np.random.default_rng(99).normal(size=rm.matrix.shape[0]) * 5
        rm.set_rf_response(new_rf)

        assert rm.rf_weight == explicit_weight


class TestSolveCacheInvalidatedAfterSetRfResponse:
    """solve() cache is invalidated after set_rf_response() so a stale
    pseudoinverse is not reused."""

    def test_cache_invalidated(self):
        rm = _make_rm(randomize_rf=True)
        output = np.random.default_rng(7).normal(size=rm.matrix.shape[0])

        # First solve to populate the cache
        result1 = rm.solve(output, method="tikhonov", parameter=1.0, rf=True)
        assert rm._inverse_RM is not None

        # Change rf_response -- this changes hash_rf_response, which
        # solve() checks before reusing the cached pseudoinverse.
        new_rf = np.random.default_rng(99).normal(size=rm.matrix.shape[0]) * 5
        rm.set_rf_response(new_rf)

        # Second solve should NOT reuse the stale pseudoinverse
        result2 = rm.solve(output, method="tikhonov", parameter=1.0, rf=True)

        # The results must differ because the rf_response (and therefore
        # the pseudoinverse) changed.
        assert not np.allclose(result1, result2)
