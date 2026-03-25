"""Tests for pySC.core.bpm_system: BPMSystem, _rotation_matrix, capture methods."""
import pytest
import numpy as np
from math import pi

from pySC.core.bpm_system import BPMSystem, _rotation_matrix


# ---------------------------------------------------------------------------
# _rotation_matrix
# ---------------------------------------------------------------------------

def test_rotation_matrix_identity():
    """_rotation_matrix(0) returns the 2x2 identity matrix."""
    R = _rotation_matrix(0)
    np.testing.assert_allclose(R, np.eye(2), atol=1e-15)


def test_rotation_matrix_90deg():
    """_rotation_matrix(pi/2) produces [[0, -1], [1, 0]]."""
    R = _rotation_matrix(pi / 2)
    expected = np.array([[0.0, -1.0], [1.0, 0.0]])
    np.testing.assert_allclose(R, expected, atol=1e-15)


# ---------------------------------------------------------------------------
# bpm_number look-up
# ---------------------------------------------------------------------------

def test_bpm_number_by_index(sc):
    """bpm_number(index=idx) returns the sequential BPM number."""
    bpm = sc.bpm_system
    idx = bpm.indices[3]
    assert bpm.bpm_number(index=idx) == 3


def test_bpm_number_by_name(sc):
    """bpm_number(name=...) returns the correct BPM number (using real HMBA names)."""
    bpm = sc.bpm_system
    # BPM names set in conftest: ring[i].FamName for monitors
    first_name = bpm.names[0]
    assert bpm.bpm_number(name=first_name) == 0
    last_name = bpm.names[-1]
    assert bpm.bpm_number(name=last_name) == len(bpm.names) - 1


def test_bpm_number_both_raises(sc):
    """Providing both index and name raises AssertionError."""
    bpm = sc.bpm_system
    with pytest.raises(AssertionError):
        bpm.bpm_number(index=bpm.indices[0], name=bpm.names[0])


def test_bpm_number_neither_raises(sc):
    """Providing neither index nor name raises AssertionError."""
    bpm = sc.bpm_system
    with pytest.raises(AssertionError):
        bpm.bpm_number()


# ---------------------------------------------------------------------------
# gain_corrections default
# ---------------------------------------------------------------------------

def test_gain_corrections_default_ones(sc):
    """After conftest initialisation, gain_corrections_x/y are all ones."""
    bpm = sc.bpm_system
    np.testing.assert_array_equal(bpm.gain_corrections_x, np.ones(len(bpm.indices)))
    np.testing.assert_array_equal(bpm.gain_corrections_y, np.ones(len(bpm.indices)))


# ---------------------------------------------------------------------------
# einsum rotation contraction
# ---------------------------------------------------------------------------

def test_einsum_rotation_explicit():
    """The einsum('ijk,jk->ik', ...) contraction matches manual per-BPM matrix multiply."""
    # Set up 2 BPMs with different rolls
    rolls = np.array([0.1, 0.3])
    rot = _rotation_matrix(rolls)  # shape (2, 2, 2)
    orbit = np.array([[1.0, 2.0],   # x for BPM 0, 1
                      [0.5, -0.3]]) # y for BPM 0, 1

    result = np.einsum('ijk,jk->ik', rot, orbit)

    # Manual: for each BPM k, result[:,k] = rot[:,:,k] @ orbit[:,k]
    for k in range(2):
        expected = rot[:, :, k] @ orbit[:, k]
        np.testing.assert_allclose(result[:, k], expected, atol=1e-15)


# ---------------------------------------------------------------------------
# capture_orbit tests (require AT tracking)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_capture_orbit_zero_errors(sc):
    """With all errors zero, capture_orbit returns the raw orbit (no BBA/ref subtraction)."""
    bpm = sc.bpm_system
    raw_orbit = sc.lattice.get_orbit(indices=bpm.indices)
    fake_x, fake_y = bpm.capture_orbit(bba=False, subtract_reference=False)
    np.testing.assert_allclose(fake_x, raw_orbit[0], atol=1e-12)
    np.testing.assert_allclose(fake_y, raw_orbit[1], atol=1e-12)


@pytest.mark.slow
def test_capture_orbit_with_offset(sc):
    """Known offset shifts the fake orbit correctly."""
    bpm = sc.bpm_system
    offset_val = 1e-4
    bpm.offsets_x[:] = offset_val
    try:
        raw_orbit = sc.lattice.get_orbit(indices=bpm.indices)
        fake_x, fake_y = bpm.capture_orbit(bba=False, subtract_reference=False)
        # formula: (raw - offset) * (1+cal) + noise; cal=0, noise=0
        expected_x = raw_orbit[0] - offset_val
        np.testing.assert_allclose(fake_x, expected_x, atol=1e-12)
        # y unchanged
        np.testing.assert_allclose(fake_y, raw_orbit[1], atol=1e-12)
    finally:
        bpm.offsets_x[:] = 0.0


@pytest.mark.slow
def test_capture_orbit_with_calibration(sc):
    """Known calibration error scales the orbit correctly."""
    bpm = sc.bpm_system
    cal_err = 0.05
    bpm.calibration_errors_x[:] = cal_err
    try:
        raw_orbit = sc.lattice.get_orbit(indices=bpm.indices)
        fake_x, _ = bpm.capture_orbit(bba=False, subtract_reference=False)
        expected_x = raw_orbit[0] * (1 + cal_err)
        np.testing.assert_allclose(fake_x, expected_x, atol=1e-12)
    finally:
        bpm.calibration_errors_x[:] = 0.0


@pytest.mark.slow
def test_capture_orbit_with_roll(sc):
    """Known roll angle rotates x/y correctly."""
    bpm = sc.bpm_system
    roll_angle = 0.1
    bpm.rolls[:] = roll_angle
    bpm.update_rot_matrices()
    try:
        raw_orbit = sc.lattice.get_orbit(indices=bpm.indices)
        fake_x, fake_y = bpm.capture_orbit(bba=False, subtract_reference=False)
        # Rotated orbit: R(roll) @ [x; y]
        c, s = np.cos(roll_angle), np.sin(roll_angle)
        expected_x = c * raw_orbit[0] - s * raw_orbit[1]
        expected_y = s * raw_orbit[0] + c * raw_orbit[1]
        np.testing.assert_allclose(fake_x, expected_x, atol=1e-12)
        np.testing.assert_allclose(fake_y, expected_y, atol=1e-12)
    finally:
        bpm.rolls[:] = 0.0
        bpm.update_rot_matrices()


@pytest.mark.slow
def test_capture_orbit_with_noise(sc):
    """With seeded RNG, noise is reproducible across two calls with same seed."""
    bpm = sc.bpm_system
    noise_level = 1e-6
    bpm.noise_co_x[:] = noise_level
    bpm.noise_co_y[:] = noise_level
    try:
        # Reset RNG to known state
        from pySC.core.rng import RNG
        sc.rng = RNG(seed=123)
        x1, y1 = bpm.capture_orbit(bba=False, subtract_reference=False)

        sc.rng = RNG(seed=123)
        x2, y2 = bpm.capture_orbit(bba=False, subtract_reference=False)

        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

        # With noise, result differs from zero-noise orbit
        bpm.noise_co_x[:] = 0.0
        bpm.noise_co_y[:] = 0.0
        sc.rng = RNG(seed=123)
        x0, y0 = bpm.capture_orbit(bba=False, subtract_reference=False)
        assert not np.allclose(x1, x0)
    finally:
        bpm.noise_co_x[:] = 0.0
        bpm.noise_co_y[:] = 0.0
        sc.rng = RNG(seed=42)


@pytest.mark.slow
def test_capture_orbit_bba_flag(sc):
    """bba=True subtracts BBA offsets, bba=False does not."""
    bpm = sc.bpm_system
    bba_val = 5e-5
    bpm.bba_offsets_x[:] = bba_val
    try:
        x_bba, _ = bpm.capture_orbit(bba=True, subtract_reference=False)
        x_no_bba, _ = bpm.capture_orbit(bba=False, subtract_reference=False)
        # bba=True subtracts bba_offsets, so x_bba = x_no_bba - bba_val
        np.testing.assert_allclose(x_bba, x_no_bba - bba_val, atol=1e-15)
    finally:
        bpm.bba_offsets_x[:] = 0.0


@pytest.mark.slow
def test_capture_orbit_subtract_reference(sc):
    """subtract_reference=True subtracts reference orbit from result."""
    bpm = sc.bpm_system
    ref_val = 3e-5
    bpm.reference_x[:] = ref_val
    try:
        x_ref, _ = bpm.capture_orbit(bba=False, subtract_reference=True)
        x_no_ref, _ = bpm.capture_orbit(bba=False, subtract_reference=False)
        np.testing.assert_allclose(x_ref, x_no_ref - ref_val, atol=1e-15)
    finally:
        bpm.reference_x[:] = 0.0


@pytest.mark.slow
def test_capture_orbit_gain_corrections(sc):
    """Setting gain_corrections_x/y multiplies the final orbit."""
    bpm = sc.bpm_system
    gain = 1.1
    bpm.gain_corrections_x[:] = gain
    try:
        x_gain, _ = bpm.capture_orbit(bba=False, subtract_reference=False)
        bpm.gain_corrections_x[:] = 1.0
        x_unity, _ = bpm.capture_orbit(bba=False, subtract_reference=False)
        # With gain, x_gain = x_unity * gain
        np.testing.assert_allclose(x_gain, x_unity * gain, atol=1e-15)
    finally:
        bpm.gain_corrections_x[:] = 1.0
        bpm.gain_corrections_y[:] = 1.0


# ---------------------------------------------------------------------------
# reconstruct_true_orbit
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_reconstruct_true_orbit_inverse(sc):
    """reconstruct(forward(x,y)) approximately recovers (x,y) when noise=0.

    Regression: reconstruct_true_orbit previously did not invert gain corrections,
    requiring callers to manually divide by gain before calling. Now it handles
    gain inversion internally.
    """
    bpm = sc.bpm_system
    # Set a small calibration error and roll so the forward path is non-trivial
    bpm.calibration_errors_x[:] = 0.02
    bpm.calibration_errors_y[:] = -0.01
    bpm.rolls[:] = 0.05
    bpm.update_rot_matrices()
    try:
        raw_orbit = sc.lattice.get_orbit(indices=bpm.indices)
        fake_x, fake_y = bpm.capture_orbit(bba=False, subtract_reference=False)

        # reconstruct_true_orbit now inverts gain internally — no manual division needed
        for k in range(len(bpm.indices)):
            rx, ry = bpm.reconstruct_true_orbit(name=bpm.names[k], x=fake_x[k], y=fake_y[k])
            np.testing.assert_allclose(rx, raw_orbit[0, k], atol=1e-10)
            np.testing.assert_allclose(ry, raw_orbit[1, k], atol=1e-10)
    finally:
        bpm.calibration_errors_x[:] = 0.0
        bpm.calibration_errors_y[:] = 0.0
        bpm.rolls[:] = 0.0
        bpm.update_rot_matrices()


# ---------------------------------------------------------------------------
# capture_injection tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_capture_injection_shape(sc):
    """Output shape is (n_bpms, n_turns) for each of x, y."""
    bpm = sc.bpm_system
    n_turns = 3
    x, y = bpm.capture_injection(n_turns=n_turns, bba=False, subtract_reference=False, use_design=True)
    assert x.shape == (len(bpm.indices), n_turns)
    assert y.shape == (len(bpm.indices), n_turns)


@pytest.mark.slow
def test_capture_injection_with_transmission(sc):
    """return_transmission=True returns a 3-tuple (x, y, transmission)."""
    bpm = sc.bpm_system
    result = bpm.capture_injection(n_turns=1, return_transmission=True, use_design=True)
    assert len(result) == 3
    x, y, transmission = result
    assert x.shape[0] == len(bpm.indices)
    assert y.shape[0] == len(bpm.indices)


@pytest.mark.slow
def test_capture_injection_design_mode(sc):
    """use_design=True uses the design lattice (no errors applied)."""
    bpm = sc.bpm_system
    x_design, y_design = bpm.capture_injection(n_turns=1, use_design=True)
    # Design-mode result should be reproducible (no RNG involved)
    x_design2, y_design2 = bpm.capture_injection(n_turns=1, use_design=True)
    np.testing.assert_array_equal(x_design, x_design2)
    np.testing.assert_array_equal(y_design, y_design2)


# ---------------------------------------------------------------------------
# capture_pseudo_orbit tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_capture_pseudo_orbit_shape(sc):
    """Output shape is (n_bpms,) for each of x, y — averaged across turns."""
    bpm = sc.bpm_system
    n_turns = 3
    x, y = bpm.capture_pseudo_orbit(n_turns=n_turns, bba=False, subtract_reference=False, use_design=True)
    assert x.shape == (len(bpm.indices),)
    assert y.shape == (len(bpm.indices),)


@pytest.mark.slow
def test_capture_pseudo_orbit_with_transmission(sc):
    """return_transmission=True returns a 3-tuple (x, y, transmission)."""
    bpm = sc.bpm_system
    result = bpm.capture_pseudo_orbit(n_turns=2, return_transmission=True, use_design=True)
    assert len(result) == 3
    x, y, transmission = result
    assert x.shape == (len(bpm.indices),)
    assert y.shape == (len(bpm.indices),)


@pytest.mark.slow
def test_capture_pseudo_orbit_single_turn(sc):
    """n_turns=1 degenerates to a squeezed single-turn injection reading."""
    bpm = sc.bpm_system
    x_pseudo, y_pseudo = bpm.capture_pseudo_orbit(n_turns=1, bba=False, subtract_reference=False, use_design=True)
    x_inj, y_inj = bpm.capture_injection(n_turns=1, bba=False, subtract_reference=False, use_design=True)
    # With use_design=True (no RNG noise), pseudo-orbit of 1 turn == injection squeezed
    np.testing.assert_allclose(x_pseudo, x_inj.squeeze())
    np.testing.assert_allclose(y_pseudo, y_inj.squeeze())
