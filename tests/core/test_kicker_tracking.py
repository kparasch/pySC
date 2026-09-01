"""Tracking hook tests for turn-by-turn kicker programs."""
from types import SimpleNamespace

import numpy as np
from pydantic import PrivateAttr

from pySC.core.lattice import ATLattice, Lattice
from pySC.core.xsuite_lattice import XSuiteLattice


class _CountingKickers:
    has_active = True

    def __init__(self):
        self.turns_applied = 0

    def apply_next_turn(self):
        self.turns_applied += 1


class _ChunkingLattice(Lattice):
    _calls: list[int] = PrivateAttr(default_factory=list)

    def track(
        self,
        bunch,
        indices=None,
        n_turns=1,
        use_design=False,
        coordinates=None,
        modify_bunch_in_place=False,
        kickers=None,
    ):
        self._calls.append(n_turns)
        n_coords = len(coordinates) if coordinates is not None else 2
        n_indices = len(indices) if indices is not None else 1
        out = np.full((n_coords, bunch.shape[0], n_indices, n_turns), np.nan)

        for turn in range(n_turns):
            if kickers is not None:
                kickers.apply_next_turn()
                out[:, :, :, turn] = kickers.turns_applied
            else:
                out[:, :, :, turn] = 0.0

        if modify_bunch_in_place:
            bunch[:, 0] += n_turns
        return out


def test_track_mean_passes_same_kickers_through_chunks():
    lattice = _ChunkingLattice(lattice_file="dummy", turns_per_chunk=2)
    bunch = np.zeros((1, 6))
    kickers = _CountingKickers()

    xy, transmission = lattice.track_mean(
        bunch,
        indices=[0],
        n_turns=5,
        kickers=kickers,
    )

    assert lattice._calls == [2, 2, 1]
    assert kickers.turns_applied == 5
    np.testing.assert_array_equal(xy[0, 0, :], np.array([1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(transmission, np.ones(5))


class _FakeATRing:
    def __init__(self):
        self.calls = []

    def track(self, r_in, refpts=None, nturns=1, in_place=True):
        self.calls.append({"refpts": refpts, "nturns": nturns, "in_place": in_place})
        r_in[0, :] = len(self.calls)
        n_refpts = len(refpts) if refpts is not None else 1
        out = np.zeros((6, r_in.shape[1], n_refpts, nturns))
        out[0, :, :, :] = len(self.calls)
        out[2, :, :, :] = 10 + len(self.calls)
        return (out,)


def test_at_lattice_track_with_kickers_uses_one_turn_loop_and_updates_bunch():
    ring = _FakeATRing()
    lattice = ATLattice.model_construct(lattice_file="dummy")
    lattice._ring = ring
    lattice._design = ring
    lattice._omp_num_threads = None
    bunch = np.zeros((1, 6))
    kickers = _CountingKickers()

    out = lattice.track(
        bunch,
        n_turns=3,
        use_design=False,
        modify_bunch_in_place=True,
        kickers=kickers,
    )

    assert [call["nturns"] for call in ring.calls] == [1, 1, 1]
    assert kickers.turns_applied == 3
    np.testing.assert_array_equal(out[0, 0, 0, :], np.array([1, 2, 3]))
    np.testing.assert_array_equal(out[1, 0, 0, :], np.array([11, 12, 13]))
    assert bunch[0, 0] == 3


class _FakeXSuiteParticles:
    def __init__(self, x, px, y, py, zeta, delta):
        self.x = np.array(x, dtype=float)
        self.px = np.array(px, dtype=float)
        self.y = np.array(y, dtype=float)
        self.py = np.array(py, dtype=float)
        self.zeta = np.array(zeta, dtype=float)
        self.delta = np.array(delta, dtype=float)
        self.state = np.ones(len(self.x), dtype=int)


class _FakeXSuiteLine:
    def __init__(self):
        self.calls = 0
        self.record_last_track = None

    def build_particles(self, x, px, y, py, zeta, delta):
        return _FakeXSuiteParticles(x, px, y, py, zeta, delta)

    def track(self, particles, turn_by_turn_monitor=None):
        self.calls += 1
        particles.x[:] = self.calls
        particles.y[:] = 10 + self.calls
        self.record_last_track = SimpleNamespace(
            state=np.ones((len(particles.x), 1), dtype=int),
            particle_id=np.arange(1, len(particles.x) + 1)[:, None],
            x=particles.x[:, None],
            px=particles.px[:, None],
            y=particles.y[:, None],
            py=particles.py[:, None],
            zeta=particles.zeta[:, None],
            delta=particles.delta[:, None],
        )


def test_xsuite_lattice_track_applies_kickers_once_per_tracked_turn():
    line = _FakeXSuiteLine()
    lattice = XSuiteLattice.model_construct(lattice_file="dummy")
    lattice._ring = line
    lattice._design = line
    bunch = np.zeros((1, 6))
    kickers = _CountingKickers()

    out = lattice.track(
        bunch,
        indices=[0],
        n_turns=3,
        use_design=False,
        modify_bunch_in_place=True,
        kickers=kickers,
    )

    assert line.calls == 3
    assert kickers.turns_applied == 3
    np.testing.assert_array_equal(out[0, 0, 0, :], np.array([1, 2, 3]))
    np.testing.assert_array_equal(out[1, 0, 0, :], np.array([11, 12, 13]))
    assert bunch[0, 0] == 3
