"""Tests for pySC.core.kickers."""
from types import SimpleNamespace

import numpy as np
import pytest

from pySC.core.kickers import (
    ACProgram,
    KickerSettings,
    SingleKickProgram,
    WhiteNoiseProgram,
)
from pySC.core.rng import RNG


class _FakeMagnetSettings:
    def __init__(self, values, seed=1):
        self.values = dict(values)
        self.calls = []
        self._parent = SimpleNamespace(rng=RNG(seed=seed))

    def get(self, control):
        return self.values[control]

    def set(self, control, value):
        self.calls.append((control, value))
        self.values[control] = value


def test_single_kick_applies_delta_relative_to_initial_value_and_finalizes():
    magnet_settings = _FakeMagnetSettings({"QF/B2": 10.0})
    program = SingleKickProgram(control="QF/B2", amplitude=0.5, turn_to_kick=1)

    program.initialize(magnet_settings)
    for _ in range(3):
        program.apply_next_turn()
    program.finalize()

    assert program._buffer == [0.0, 0.5, 0.0]
    assert magnet_settings.calls == [
        ("QF/B2", 10.0),
        ("QF/B2", 10.5),
        ("QF/B2", 10.0),
        ("QF/B2", 10.0),
    ]


def test_ac_program_generates_ramp_flat_top_and_ramp_down_values():
    program = ACProgram(
        control="QF/B2",
        amplitude=2.0,
        tune=0.25,
        ramp_up_turns=2,
        flat_top_turns=2,
        ramp_down_turns=2,
    )

    generator = program.get_generator()
    values = [next(generator) for _ in range(7)]

    np.testing.assert_allclose(values, [0.0, 1.0, 0.0, -2.0, 0.0, 1.0, 0.0], atol=1e-15)


def test_white_noise_program_uses_parent_rng():
    magnet_settings = _FakeMagnetSettings({"SF/B1": 1.0}, seed=123)
    expected_rng = RNG(seed=123)
    expected_values = [1.0 + expected_rng.normal() * 0.2 for _ in range(3)]
    program = WhiteNoiseProgram(control="SF/B1", amplitude=0.2)

    program.initialize(magnet_settings)
    for _ in range(3):
        program.apply_next_turn()

    observed_values = [value for _, value in magnet_settings.calls]
    np.testing.assert_allclose(observed_values, expected_values)


def test_kicker_settings_activate_rejects_two_active_programs_on_same_control():
    kickers = KickerSettings(
        programs={
            "first": SingleKickProgram(control="SF/B1", amplitude=1.0, turn_to_kick=0),
            "second": WhiteNoiseProgram(control="SF/B1", amplitude=0.1),
        }
    )

    kickers.activate("first")

    with pytest.raises(ValueError, match="control SF/B1 is already used"):
        kickers.activate("second")

    assert kickers.active_programs == ["first"]


def test_kicker_settings_lifecycle_only_runs_active_programs():
    magnet_settings = _FakeMagnetSettings({"SF/B1": 0.0, "SF/A1": 1.0})
    kickers = KickerSettings(
        programs={
            "active": SingleKickProgram(control="SF/B1", amplitude=2.0, turn_to_kick=0),
            "inactive": SingleKickProgram(control="SF/A1", amplitude=3.0, turn_to_kick=0),
        }
    )
    kickers.activate("active")

    kickers.initialize(magnet_settings)
    kickers.apply_next_turn()
    kickers.finalize()

    assert magnet_settings.calls == [
        ("SF/B1", 2.0),
        ("SF/B1", 0.0),
    ]
    assert kickers.programs["inactive"]._generator is None
