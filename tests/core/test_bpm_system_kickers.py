"""BPM capture tests for turn-by-turn kicker lifecycle wiring."""
from types import SimpleNamespace

import numpy as np

from pySC.core.bpm_system import BPMSystem
from pySC.core.rng import RNG


class _FakeKickers:
    has_active = True

    def __init__(self):
        self.initialized_with = []
        self.finalize_count = 0

    def initialize(self, magnet_settings):
        self.initialized_with.append(magnet_settings)

    def finalize(self):
        self.finalize_count += 1


class _FakeInjection:
    def generate_bunch(self, use_design=False):
        return np.zeros((1, 6))

    def generate_orbit_centered_bunch(self, use_design=False):
        return np.zeros((1, 6))


class _FakeLattice:
    def __init__(self):
        self.calls = []

    def track_mean(
        self,
        bunch,
        indices=None,
        n_turns=1,
        use_design=False,
        coordinates=None,
        transmission_threshold=0,
        kickers=None,
    ):
        self.calls.append(
            {
                "use_design": use_design,
                "n_turns": n_turns,
                "kickers": kickers,
            }
        )
        trajectory = np.zeros((2, len(indices), n_turns))
        transmission = np.ones(n_turns)
        return trajectory, transmission


def _make_bpm_system(parent):
    bpm_system = BPMSystem(indices=[0], names=["BPM"])
    bpm_system._parent = parent
    bpm_system.calibration_errors_x = np.zeros(1)
    bpm_system.calibration_errors_y = np.zeros(1)
    bpm_system.noise_tbt_x = np.zeros(1)
    bpm_system.noise_tbt_y = np.zeros(1)
    bpm_system.update_rot_matrices()
    return bpm_system


def test_capture_injection_initializes_design_kickers_and_finalizes():
    kickers = _FakeKickers()
    lattice = _FakeLattice()
    design_magnet_settings = object()
    parent = SimpleNamespace(
        kickers=kickers,
        design_magnet_settings=design_magnet_settings,
        magnet_settings=object(),
        injection=_FakeInjection(),
        lattice=lattice,
        rng=RNG(seed=1),
    )
    bpm_system = _make_bpm_system(parent)

    bpm_system.capture_injection(n_turns=3, use_design=True)

    assert kickers.initialized_with == [design_magnet_settings]
    assert kickers.finalize_count == 1
    assert lattice.calls[0]["kickers"] is kickers
    assert lattice.calls[0]["use_design"] is True


def test_capture_kick_initializes_ring_kickers_and_finalizes():
    kickers = _FakeKickers()
    lattice = _FakeLattice()
    magnet_settings = object()
    parent = SimpleNamespace(
        kickers=kickers,
        design_magnet_settings=object(),
        magnet_settings=magnet_settings,
        injection=_FakeInjection(),
        lattice=lattice,
        rng=RNG(seed=1),
    )
    bpm_system = _make_bpm_system(parent)

    bpm_system.capture_kick(n_turns=3, use_design=False, bba=False, subtract_reference=False)

    assert kickers.initialized_with == [magnet_settings]
    assert kickers.finalize_count == 1
    assert lattice.calls[0]["kickers"] is kickers
    assert lattice.calls[0]["use_design"] is False
