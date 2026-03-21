"""Tests for pySC.core.lattice: ATLattice loading, tracking, optics, magnets."""
import pytest
import numpy as np
import at

from pySC.core.lattice import ATLattice


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Lattice construction
# ---------------------------------------------------------------------------

def test_lattice_loads_from_file(hmba_lattice_file):
    """ATLattice loads from a .mat file and exposes ring/design AT lattices."""
    lat = ATLattice(lattice_file=hmba_lattice_file)
    assert lat.ring is not None
    assert lat.design is not None
    assert isinstance(lat.ring, at.Lattice)
    assert isinstance(lat.design, at.Lattice)
    assert len(lat.ring) > 0
    assert len(lat.design) == len(lat.ring)


# ---------------------------------------------------------------------------
# Orbit
# ---------------------------------------------------------------------------

def test_get_orbit_returns_xy(hmba_lattice_file):
    """get_orbit() returns a 2-row numpy array (x, y) for all elements."""
    lat = ATLattice(lattice_file=hmba_lattice_file)
    orbit = lat.get_orbit(use_design=True)
    assert isinstance(orbit, np.ndarray)
    assert orbit.shape[0] == 2  # x and y rows
    assert orbit.shape[1] == len(lat.design)


def test_get_orbit_design_vs_perturbed(hmba_lattice_file):
    """Design orbit differs from perturbed orbit after a magnet kick."""
    lat = ATLattice(lattice_file=hmba_lattice_file)
    orbit_design = lat.get_orbit(use_design=True).copy()

    # Find a quadrupole and apply a small gradient perturbation on the ring
    ring = lat.ring
    quad_idx = next(i for i, e in enumerate(ring) if isinstance(e, at.Quadrupole))
    ring[quad_idx].PolynomB[1] += 0.001  # small perturbation

    orbit_perturbed = lat.get_orbit(use_design=False)
    # The x orbits should now differ
    assert not np.allclose(orbit_design[0], orbit_perturbed[0], atol=1e-12), (
        "Orbit should change after perturbing a quadrupole gradient"
    )

    # Restore
    ring[quad_idx].PolynomB[1] -= 0.001


# ---------------------------------------------------------------------------
# Twiss
# ---------------------------------------------------------------------------

def test_get_twiss_keys(hmba_lattice_file):
    """get_twiss() returns a dict with expected optics keys."""
    lat = ATLattice(lattice_file=hmba_lattice_file)
    twiss = lat.get_twiss(use_design=True)
    expected_keys = {'qx', 'qy', 'qs', 'dqx', 'dqy', 's', 'x', 'px', 'y', 'py',
                     'delta', 'tau', 'betx', 'bety', 'alfx', 'alfy', 'mux', 'muy',
                     'dx', 'dpx', 'dy', 'dpy'}
    assert expected_keys.issubset(twiss.keys())


# ---------------------------------------------------------------------------
# Tune and chromaticity
# ---------------------------------------------------------------------------

def test_get_tune(hmba_lattice_file):
    """get_tune() returns two fractional tunes in (0, 1)."""
    lat = ATLattice(lattice_file=hmba_lattice_file)
    qx, qy = lat.get_tune(use_design=True)
    assert isinstance(qx, (float, np.floating))
    assert isinstance(qy, (float, np.floating))
    assert 0 < qx < 1
    assert 0 < qy < 1


def test_get_chromaticity(hmba_lattice_file):
    """get_chromaticity() returns two finite floats."""
    lat = ATLattice(lattice_file=hmba_lattice_file)
    dqx, dqy = lat.get_chromaticity(use_design=True)
    assert isinstance(dqx, (float, np.floating))
    assert isinstance(dqy, (float, np.floating))
    assert np.isfinite(dqx)
    assert np.isfinite(dqy)


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def test_track_single_particle(sc):
    """track() returns array shaped (coords, particles, refpts, turns)."""
    bunch = np.zeros((1, 6))
    n_turns = 10
    result = sc.lattice.track(bunch, n_turns=n_turns, use_design=True)
    # Default coordinates = ['x', 'y'], 1 particle, no refpts -> 1 refpt, 10 turns
    assert result.shape[0] == 2  # x, y
    assert result.shape[1] == 1  # 1 particle
    assert result.shape[3] == n_turns


def test_track_mean_shape(sc):
    """track_mean() output has shape (coords, indices, turns)."""
    bunch = np.zeros((3, 6))  # 3 particles
    bpm_indices = sc.bpm_system.indices
    n_turns = 5
    xy, transmission = sc.lattice.track_mean(
        bunch, indices=bpm_indices, n_turns=n_turns, use_design=True
    )
    assert xy.shape == (2, len(bpm_indices), n_turns)
    assert transmission.shape == (n_turns,)


def test_track_mean_chunked(sc):
    """track_mean with small turns_per_chunk matches large turns_per_chunk result."""
    bunch = np.zeros((2, 6))
    bpm_indices = sc.bpm_system.indices[:3]  # use a few BPMs for speed
    n_turns = 12

    # Large chunk (single pass)
    original_chunk = sc.lattice.turns_per_chunk
    sc.lattice.turns_per_chunk = n_turns + 1  # larger than n_turns
    xy_single, trans_single = sc.lattice.track_mean(
        bunch, indices=bpm_indices, n_turns=n_turns, use_design=True
    )

    # Small chunk (multiple passes)
    sc.lattice.turns_per_chunk = 4  # 12 / 4 = 3 chunks exactly
    xy_chunked, trans_chunked = sc.lattice.track_mean(
        bunch, indices=bpm_indices, n_turns=n_turns, use_design=True
    )

    # Restore
    sc.lattice.turns_per_chunk = original_chunk

    np.testing.assert_allclose(xy_single, xy_chunked, atol=1e-12)
    np.testing.assert_allclose(trans_single, trans_chunked, atol=1e-12)


# ---------------------------------------------------------------------------
# Magnet components
# ---------------------------------------------------------------------------

def test_set_get_magnet_component_roundtrip(sc):
    """set_magnet_component then get_magnet_component round-trips the value."""
    ring = sc.lattice.design
    quad_idx = next(i for i, e in enumerate(ring) if isinstance(e, at.Quadrupole))

    original = sc.lattice.get_magnet_component(quad_idx, 'B', order=1, use_design=True)
    test_value = original + 0.123
    sc.lattice.set_magnet_component(quad_idx, test_value, 'B', order=1, use_design=True)
    readback = sc.lattice.get_magnet_component(quad_idx, 'B', order=1, use_design=True)
    assert readback == pytest.approx(test_value)

    # Restore
    sc.lattice.set_magnet_component(quad_idx, original, 'B', order=1, use_design=True)


def test_corrector_kickangle_convention(hmba_lattice_file):
    """For at.Corrector, B component maps to negative KickAngle[0]."""
    lat = ATLattice(lattice_file=hmba_lattice_file)

    # Insert a corrector into the ring for testing
    cor = at.Corrector('TESTCOR', 0.1, [0.0, 0.0])
    ring = lat.ring
    ring.insert(1, cor)
    design = lat.design
    design.insert(1, cor)

    idx = 1  # where we just inserted
    test_val = 0.005
    length = lat.get_length(idx)
    lat.set_magnet_component(idx, test_val, 'B', order=0, use_design=False)

    # B component -> KickAngle[0] = -value * length
    assert ring[idx].KickAngle[0] == pytest.approx(-test_val * length)

    readback = lat.get_magnet_component(idx, 'B', order=0, use_design=False)
    assert readback == pytest.approx(test_val)


# ---------------------------------------------------------------------------
# RF cavity
# ---------------------------------------------------------------------------

def test_update_cavity_voltage_phase_frequency(sc):
    """update_cavity then get_cavity_voltage_phase_frequency round-trips."""
    ring = sc.lattice.design
    rf_idx = next(i for i, e in enumerate(ring) if isinstance(e, at.RFCavity))

    voltage_orig, phase_orig, freq_orig = sc.lattice.get_cavity_voltage_phase_frequency(rf_idx)

    new_voltage = voltage_orig * 1.1
    new_phase = 10.0
    new_freq = freq_orig + 100.0

    sc.lattice.update_cavity(rf_idx, new_voltage, new_phase, new_freq)
    v, p, f = sc.lattice.get_cavity_voltage_phase_frequency(rf_idx)

    assert v == pytest.approx(new_voltage)
    assert p == pytest.approx(new_phase, abs=1e-6)
    assert f == pytest.approx(new_freq)

    # Restore
    sc.lattice.update_cavity(rf_idx, voltage_orig, phase_orig, freq_orig)


# ---------------------------------------------------------------------------
# Element search and classification
# ---------------------------------------------------------------------------

def test_find_with_regex(sc):
    """find_with_regex('QF') returns indices of elements whose FamName matches."""
    indices = sc.lattice.find_with_regex("QF")
    assert len(indices) > 0
    for idx in indices:
        assert "QF" in sc.lattice.ring[idx].FamName


def test_is_dipole(sc):
    """Dipole with BendingAngle returns True, drift returns False."""
    ring = sc.lattice.design
    dip_idx = next(i for i, e in enumerate(ring)
                   if hasattr(e, 'BendingAngle') and e.BendingAngle != 0)
    drift_idx = next(i for i, e in enumerate(ring) if isinstance(e, at.Drift))

    assert sc.lattice.is_dipole(dip_idx) is True
    assert sc.lattice.is_dipole(drift_idx) is False


# ---------------------------------------------------------------------------
# Misalignment
# ---------------------------------------------------------------------------

def test_update_misalignment(sc):
    """update_misalignment(dx=0.001) modifies element T1/T2/R1/R2."""
    ring = sc.lattice.ring
    quad_idx = next(i for i, e in enumerate(ring) if isinstance(e, at.Quadrupole))
    elem = ring[quad_idx]

    # Ensure T1 exists and record baseline
    if not hasattr(elem, 'T1'):
        elem.T1 = np.zeros(6)
    t1_before = elem.T1.copy()

    sc.lattice.update_misalignment(quad_idx, dx=0.001)

    assert hasattr(elem, 'T1')
    assert hasattr(elem, 'T2')
    assert hasattr(elem, 'R1')
    assert hasattr(elem, 'R2')
    assert not np.allclose(elem.T1, t1_before), "T1 should change after misalignment"


# ---------------------------------------------------------------------------
# One-turn matrix
# ---------------------------------------------------------------------------

def test_one_turn_matrix_shape(hmba_lattice_file):
    """one_turn_matrix returns 6x6 (6d) or 4x4 (no_6d)."""
    # 6D lattice
    lat_6d = ATLattice(lattice_file=hmba_lattice_file, no_6d=False)
    M6 = lat_6d.one_turn_matrix(use_design=True)
    assert M6.shape == (6, 6)

    # 4D lattice
    lat_4d = ATLattice(lattice_file=hmba_lattice_file, no_6d=True)
    M4 = lat_4d.one_turn_matrix(use_design=True)
    assert M4.shape == (4, 4)


# ---------------------------------------------------------------------------
# Coordinate swap convention
# ---------------------------------------------------------------------------

def test_coordinate_swap(hmba_lattice_file):
    """pySC swaps columns 4 and 5 of the input bunch before passing to AT.

    In pySC convention the user places delta (energy deviation) at index 4
    and tau (path length) at index 5 of the bunch array. Before calling AT,
    pySC swaps these two columns so that AT receives dp at its native
    index 4 and ct at its native index 5.

    On output, pySC's coord_map maps 'delta'->4, 'tau'->5, reading AT's
    native output indices directly. This test verifies that a particle
    with a known energy deviation placed at pySC input index 4 produces
    the expected behaviour: the delta coordinate in the output (AT index 4)
    reflects the energy deviation, while tau (AT index 5) is nonzero due
    to path-length accumulation.
    """
    lat = ATLattice(lattice_file=hmba_lattice_file)

    # Place a small energy deviation at pySC index 4 (delta).
    # After the input swap this becomes AT ct (index 5), so AT sees dp=0.
    # But a separate particle with pySC index 5 (tau) should map to AT dp.
    #
    # Strategy: create two particles -- one with value at pySC[4] and
    # one with value at pySC[5] -- and verify they map to different AT
    # coordinates by checking the output.
    delta_val = 1e-4
    bunch_a = np.zeros((1, 6))
    bunch_a[0, 4] = delta_val  # pySC delta -> after swap -> AT index 5 (ct)

    bunch_b = np.zeros((1, 6))
    bunch_b[0, 5] = delta_val  # pySC tau -> after swap -> AT index 4 (dp)

    result_a = lat.track(
        bunch_a, n_turns=1, use_design=True,
        coordinates=['x', 'px', 'y', 'py', 'delta', 'tau']
    )
    result_b = lat.track(
        bunch_b, n_turns=1, use_design=True,
        coordinates=['x', 'px', 'y', 'py', 'delta', 'tau']
    )

    # bunch_b has energy offset (dp != 0 in AT), so 'delta' output should
    # show it approximately preserved after one turn (radiation loss causes
    # some change, so use a generous relative tolerance)
    delta_out_b = result_b[4, 0, 0, 0]  # 'delta' -> AT index 4 (dp)
    assert delta_out_b == pytest.approx(delta_val, rel=0.2), (
        f"Energy deviation via pySC[5] should appear in 'delta' output, got {delta_out_b}"
    )

    # bunch_a has ct offset (pySC[4] -> AT[5] ct), not dp, so 'delta'
    # output should NOT be delta_val
    delta_out_a = result_a[4, 0, 0, 0]
    assert not np.isclose(delta_out_a, delta_val, rtol=0.1), (
        "pySC input index 4 should be swapped to AT ct, not dp"
    )
