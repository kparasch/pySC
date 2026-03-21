"""Tests for pySC.utils.rdt: Resonance Driving Terms and related functions."""
import numpy as np
import pytest

from pySC.utils.rdt import (
    binomial_coeff,
    feeddown,
    omega,
    FACTORIAL,
    S4,
    linear_normal_form,
    Rot2D,
    hjklm,
    fjklm,
)


class TestBinomialCoeff:

    def test_binomial_known_values(self):
        """binomial_coeff reproduces standard binomial coefficients."""
        assert binomial_coeff(5, 0) == pytest.approx(1)
        assert binomial_coeff(5, 1) == pytest.approx(5)
        assert binomial_coeff(5, 2) == pytest.approx(10)
        assert binomial_coeff(5, 3) == pytest.approx(10)
        assert binomial_coeff(5, 5) == pytest.approx(1)
        assert binomial_coeff(10, 3) == pytest.approx(120)

    def test_binomial_symmetry(self):
        """C(n,k) == C(n, n-k)."""
        for n in range(1, 12):
            for k in range(n + 1):
                assert binomial_coeff(n, k) == pytest.approx(binomial_coeff(n, n - k))


class TestOmega:

    def test_omega_even(self):
        """omega(x) returns 1 for even x."""
        assert omega(0) == 1
        assert omega(2) == 1
        assert omega(4) == 1

    def test_omega_odd(self):
        """omega(x) returns 0 for odd x."""
        assert omega(1) == 0
        assert omega(3) == 0
        assert omega(5) == 0


class TestFeeddown:

    def test_feeddown_no_offset(self):
        """With r0=0, feeddown returns AB[n] (no contributions from higher orders)."""
        AB = np.array([0.1 + 0.2j, 0.3 + 0.4j, 0.5 + 0.6j])
        for n in range(len(AB)):
            result = feeddown(AB, r0=0.0, n=n)
            assert result == pytest.approx(AB[n], abs=1e-14)

    def test_feeddown_with_offset(self):
        """feeddown with non-zero r0 includes higher-order contributions."""
        # For a pure quadrupole: AB = [0, K] (K at index 1)
        K = 1.0 + 0.0j
        AB = np.array([0.0 + 0.0j, K])
        r0 = 0.001 + 0.0j  # 1 mm horizontal offset

        # feeddown(AB, r0, n=0) = AB[0]*C(0,0)*r0^0 + AB[1]*C(1,0)*r0^1
        #                        = 0 + K * 1 * r0
        result = feeddown(AB, r0, n=0)
        expected = K * r0
        assert result == pytest.approx(expected, abs=1e-14)

        # feeddown(AB, r0, n=1) = AB[1]*C(1,1)*r0^0 = K
        result = feeddown(AB, r0, n=1)
        assert result == pytest.approx(K, abs=1e-14)

    def test_feeddown_sextupole(self):
        """feeddown for a sextupole (order 2) with offset gives a linear feed-down."""
        # AB = [0, 0, S] — pure sextupole at index 2
        S = 2.0 + 0.0j
        AB = np.array([0.0j, 0.0j, S])
        r0 = 0.002 + 0.001j

        # Feed-down to n=1: S * C(2,1) * r0^1 = S * 2 * r0
        result = feeddown(AB, r0, n=1)
        expected = S * 2 * r0
        assert result == pytest.approx(expected, abs=1e-14)

        # Feed-down to n=0: S * C(2,0) * r0^2 = S * r0^2
        result = feeddown(AB, r0, n=0)
        expected = S * r0**2
        assert result == pytest.approx(expected, abs=1e-14)


class TestRot2D:

    def test_rot2d_identity(self):
        """Rot2D(0) is the 2x2 identity."""
        np.testing.assert_allclose(Rot2D(0), np.eye(2), atol=1e-15)

    def test_rot2d_orthogonal(self):
        """Rot2D always produces an orthogonal matrix (R^T R = I)."""
        mu = 1.234
        R = Rot2D(mu)
        np.testing.assert_allclose(R.T @ R, np.eye(2), atol=1e-14)

    def test_rot2d_composition(self):
        """Rot2D(a) @ Rot2D(b) == Rot2D(a+b)."""
        a, b = 0.3, 0.7
        np.testing.assert_allclose(Rot2D(a) @ Rot2D(b), Rot2D(a + b), atol=1e-14)


class TestLinearNormalForm:

    def test_symplectic_identity(self):
        """linear_normal_form of an uncoupled map recovers known tunes and symplectic W."""
        # Build a simple stable one-turn map: uncoupled with known tunes
        mux = 0.31 * 2 * np.pi
        muy = 0.22 * 2 * np.pi
        M = np.zeros((4, 4))
        M[0:2, 0:2] = Rot2D(mux)
        M[2:4, 2:4] = Rot2D(muy)
        W, invW, R, q1, q2 = linear_normal_form(M)

        # Tunes should match
        assert abs(q1) == pytest.approx(0.31, abs=1e-10)
        assert abs(q2) == pytest.approx(0.22, abs=1e-10)

        # W should be symplectic: W^T S4 W = S4
        np.testing.assert_allclose(W.T @ S4 @ W, S4, atol=1e-12)

        # R should be block-diagonal rotation
        np.testing.assert_allclose(R[0:2, 0:2], Rot2D(2 * np.pi * q1), atol=1e-12)
        np.testing.assert_allclose(R[2:4, 2:4], Rot2D(2 * np.pi * q2), atol=1e-12)

    def test_inverse_relation(self):
        """invW should satisfy invW @ W = I."""
        mux = 0.25 * 2 * np.pi
        muy = 0.35 * 2 * np.pi
        M = np.zeros((4, 4))
        M[0:2, 0:2] = Rot2D(mux)
        M[2:4, 2:4] = Rot2D(muy)
        W, invW, R, q1, q2 = linear_normal_form(M)

        np.testing.assert_allclose(invW @ W, np.eye(4), atol=1e-12)

    def test_coupled_map(self):
        """linear_normal_form handles a weakly coupled symplectic transfer map."""
        # Build a properly symplectic coupled map using thin-lens coupling kick
        mux = 0.28 * 2 * np.pi
        muy = 0.19 * 2 * np.pi

        # Uncoupled rotation
        M0 = np.zeros((4, 4))
        M0[0:2, 0:2] = Rot2D(mux)
        M0[2:4, 2:4] = Rot2D(muy)

        # Symplectic coupling kick (skew quad thin lens): preserves symplecticity
        C = np.eye(4)
        kappa = 0.01  # weak skew quad
        C[1, 2] = kappa
        C[3, 0] = kappa

        M = M0 @ C  # coupled one-turn map

        W, invW, R, q1, q2 = linear_normal_form(M)

        # invW @ W should still be identity
        np.testing.assert_allclose(invW @ W, np.eye(4), atol=1e-10)
        # Tunes should be close to uncoupled values (weak coupling)
        assert abs(q1) == pytest.approx(0.28, abs=0.02)
        assert abs(q2) == pytest.approx(0.19, abs=0.02)
        # Off-diagonal blocks of W should be small but non-zero (coupling present)
        coupling_block = np.sqrt(W[2, 0]**2 + W[2, 1]**2)
        assert coupling_block > 1e-6, "Coupling should produce non-zero off-diagonal W elements"


class TestFactorial:

    def test_factorial_values(self):
        """FACTORIAL lookup table matches known factorials."""
        import math
        for n in range(len(FACTORIAL)):
            assert FACTORIAL[n] == math.factorial(n)


class TestS4:

    def test_s4_antisymmetric(self):
        """S4 is antisymmetric (S4^T = -S4) and S4^2 = -I."""
        np.testing.assert_array_equal(S4.T, -S4)
        np.testing.assert_allclose(S4 @ S4, -np.eye(4), atol=1e-15)


def _make_synthetic_twiss(n_elements=20, qx=0.31, qy=0.22):
    """Build a synthetic twiss dict for testing hjklm/fjklm without a real lattice."""
    s = np.linspace(0, 100, n_elements + 1)
    betx = 10 + 5 * np.sin(2 * np.pi * s / 100)
    bety = 8 + 3 * np.cos(2 * np.pi * s / 100)
    mux = np.linspace(0, qx, n_elements + 1)
    muy = np.linspace(0, qy, n_elements + 1)
    return {
        's': s,
        'qx': qx,
        'qy': qy,
        'betx': betx,
        'bety': bety,
        'mux': mux,
        'muy': muy,
        'x': np.zeros(n_elements + 1),
        'y': np.zeros(n_elements + 1),
    }


def _make_synthetic_strengths(n_elements=20, sext_indices=None, sext_k2l=50.0):
    """Build synthetic integrated strengths with sextupoles at given indices."""
    N = n_elements + 1
    if sext_indices is None:
        sext_indices = [5, 15]

    strengths = {
        'norm': {0: np.zeros(N), 1: np.zeros(N), 2: np.zeros(N)},
        'skew': {0: np.zeros(N), 1: np.zeros(N), 2: np.zeros(N)},
    }
    # Place sextupole normal strengths at order 2 (n-1=2 means j+k+l+m=3)
    for idx in sext_indices:
        strengths['norm'][2][idx] = sext_k2l

    return strengths


class TestHjklm:
    """Tests for hjklm using synthetic twiss and integrated strengths."""

    def test_hjklm_returns_correct_shape(self):
        """hjklm returns an array of the same length as twiss['s']."""
        twiss = _make_synthetic_twiss()
        strengths = _make_synthetic_strengths()
        h = hjklm(j=2, k=1, l=0, m=0, integrated_strengths=strengths, twiss=twiss)
        assert len(h) == len(twiss['s'])
        assert h.dtype == complex

    def test_hjklm_nonzero_at_sextupoles(self):
        """hjklm (h21000) is non-zero only where integrated strengths are non-zero."""
        twiss = _make_synthetic_twiss()
        sext_indices = [5, 15]
        strengths = _make_synthetic_strengths(sext_indices=sext_indices)
        h = hjklm(j=2, k=1, l=0, m=0, integrated_strengths=strengths, twiss=twiss)

        # h should be non-zero at sextupole locations
        for idx in sext_indices:
            assert h[idx] != 0, f"h21000 should be non-zero at sextupole index {idx}"

        # h should be zero where there are no strengths
        non_sext = [i for i in range(len(twiss['s'])) if i not in sext_indices]
        for idx in non_sext:
            assert h[idx] == 0, f"h21000 should be zero at non-sextupole index {idx}"

    def test_hjklm_skew_term(self):
        """hjklm for a skew-sensitive term (l+m odd) uses skew strengths."""
        twiss = _make_synthetic_twiss()
        N = len(twiss['s'])
        strengths = {
            'norm': {0: np.zeros(N), 1: np.zeros(N), 2: np.zeros(N)},
            'skew': {0: np.zeros(N), 1: np.zeros(N), 2: np.zeros(N)},
        }
        # Place a skew sextupole strength
        strengths['skew'][2][7] = 30.0

        # h10110: j=1, k=0, l=1, m=1 -> n=3, l+m=2 (even) -> uses norm via omega(l+m)=1
        # h10200: j=1, k=0, l=2, m=0 -> n=3, l+m=2 (even) -> uses norm via omega(l+m)=1
        # h10011: j=1, k=0, l=0, m=2 -> n=3, l+m=2 (even) -> norm
        # To test skew: l+m+1 must be even, i.e., l+m must be odd
        # h10100: j=1, k=0, l=1, m=0 -> n=2, but we need n>0 and strengths at n-1
        # Actually: for l+m odd, omega(l+m)=0 and omega(l+m+1)=1, so the skew component contributes.
        # h20010: j=2, k=0, l=0, m=1 -> n=3, l+m=1 (odd) -> omega(1)=0 for norm, omega(2)=1 for skew
        h = hjklm(j=2, k=0, l=0, m=1, integrated_strengths=strengths, twiss=twiss)
        assert h[7] != 0, "Skew sextupole should drive h20010"

    def test_hjklm_zero_strengths(self):
        """hjklm returns all zeros when integrated strengths are zero."""
        twiss = _make_synthetic_twiss()
        N = len(twiss['s'])
        strengths = {
            'norm': {0: np.zeros(N), 1: np.zeros(N), 2: np.zeros(N)},
            'skew': {0: np.zeros(N), 1: np.zeros(N), 2: np.zeros(N)},
        }
        h = hjklm(j=2, k=1, l=0, m=0, integrated_strengths=strengths, twiss=twiss)
        np.testing.assert_array_equal(h, np.zeros(N, dtype=complex))


class TestFjklm:
    """Tests for fjklm using synthetic data."""

    def test_fjklm_returns_correct_shape(self):
        """fjklm returns an array of the same length as twiss['s']."""
        twiss = _make_synthetic_twiss()
        strengths = _make_synthetic_strengths()
        f = fjklm(j=2, k=1, l=0, m=0, integrated_strengths=strengths, twiss=twiss)
        assert len(f) == len(twiss['s'])
        assert f.dtype == complex

    def test_fjklm_normalized_vs_unnormalized(self):
        """Normalized fjklm equals unnormalized divided by the resonance denominator."""
        twiss = _make_synthetic_twiss()
        strengths = _make_synthetic_strengths()

        f_norm = fjklm(j=2, k=1, l=0, m=0, integrated_strengths=strengths,
                       twiss=twiss, normalized=True)
        f_raw = fjklm(j=2, k=1, l=0, m=0, integrated_strengths=strengths,
                      twiss=twiss, normalized=False)

        qx = twiss["qx"]
        qy = twiss["qy"]
        # j=2, k=1, l=0, m=0 -> denominator uses (j-k)*qx + (l-m)*qy = 1*qx
        denom = 1 - np.exp(1j * 2 * np.pi * ((2 - 1) * qx + (0 - 0) * qy))
        np.testing.assert_allclose(f_norm, f_raw / denom, atol=1e-10)

    def test_fjklm_nonzero_with_sextupoles(self):
        """fjklm is non-zero when sextupole strengths are present."""
        twiss = _make_synthetic_twiss()
        strengths = _make_synthetic_strengths()
        f = fjklm(j=2, k=1, l=0, m=0, integrated_strengths=strengths, twiss=twiss)
        assert np.any(f != 0), "f21000 should be non-zero with sextupole strengths"

    def test_fjklm_zero_without_strengths(self):
        """fjklm returns all zeros when there are no magnet strengths."""
        twiss = _make_synthetic_twiss()
        N = len(twiss['s'])
        strengths = {
            'norm': {0: np.zeros(N), 1: np.zeros(N), 2: np.zeros(N)},
            'skew': {0: np.zeros(N), 1: np.zeros(N), 2: np.zeros(N)},
        }
        f = fjklm(j=2, k=1, l=0, m=0, integrated_strengths=strengths, twiss=twiss)
        np.testing.assert_array_equal(f, np.zeros(N, dtype=complex))

    @pytest.mark.regression
    def test_fjklm_single_sextupole_analytic(self):
        """Compare fjklm against the analytic RDT for a single thin sextupole.

        For a single sextupole at index s with integrated strength K2L,
        the driving term h21000 at that element is:
          h_s = -K2L / (8 * 1) * betx_s^(3/2)
        and the RDT at observation point i (unnormalized) is:
          f_i = h_s * exp(i * (j-k) * dphi_x)
        where dphi_x is the (wrapped) phase advance from s to i.

        Regression: the original fjklm used np.abs on phase differences,
        which destroyed the sign needed for correct wrapping.
        """
        N = 20
        qx = 0.31
        qy = 0.22
        sext_idx = 7
        K2L = 50.0

        twiss = _make_synthetic_twiss(n_elements=N, qx=qx, qy=qy)
        strengths = _make_synthetic_strengths(n_elements=N, sext_indices=[sext_idx], sext_k2l=K2L)

        # Compute fjklm (unnormalized) for h21000: j=2, k=1, l=0, m=0
        f = fjklm(j=2, k=1, l=0, m=0, integrated_strengths=strengths,
                  twiss=twiss, normalized=False)

        # Analytic: h21000 at sextupole location
        # h = -K2L / (2! * 1! * 0! * 0! * 2^3) * betx^(3/2)
        #   = -K2L / (2 * 1 * 1 * 1 * 8) * betx^(3/2)
        #   = -K2L / 16 * betx^(3/2)
        betx_s = twiss['betx'][sext_idx]
        h_analytic = -K2L / 16 * betx_s**(3./2)

        for ii in range(N + 1):
            dphi = 2 * np.pi * (twiss['mux'][ii] - twiss['mux'][sext_idx])
            if dphi < 0:
                dphi += 2 * np.pi * qx
            # j-k = 1 for h21000
            f_analytic = h_analytic * np.exp(1j * dphi)
            np.testing.assert_allclose(f[ii], f_analytic, atol=1e-10,
                                       err_msg=f"Mismatch at element {ii}")
