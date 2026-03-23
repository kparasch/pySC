import numpy as np
import pytest
from pySC.apps.response_matrix import ResponseMatrix


def _make_rm(n_outputs=10, n_inputs=6, rf_response=None, rf_weight=None):
    """Helper to build a minimal ResponseMatrix for testing."""
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(n_outputs, n_inputs))
    if rf_response is None:
        rf_response = rng.normal(size=n_outputs)
    kwargs = dict(matrix=matrix, rf_response=rf_response)
    if rf_weight is not None:
        kwargs["rf_weight"] = rf_weight
    return ResponseMatrix(**kwargs)


class TestSetRfResponseRecomputesWeight:
    """set_rf_response() recomputes rf_weight when it was auto-computed."""

    def test_auto_weight_is_recomputed(self):
        rm = _make_rm()
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
        rm = _make_rm(rf_weight=explicit_weight)
        assert rm._rf_weight_is_default is False
        assert rm.rf_weight == explicit_weight

        new_rf = np.random.default_rng(99).normal(size=rm.matrix.shape[0]) * 5
        rm.set_rf_response(new_rf)

        assert rm.rf_weight == explicit_weight


class TestSolveCacheInvalidatedAfterSetRfResponse:
    """solve() cache is invalidated after set_rf_response() so a stale
    pseudoinverse is not reused."""

    def test_cache_invalidated(self):
        rm = _make_rm()
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
