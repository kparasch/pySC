from pydantic import BaseModel, PrivateAttr, Field
from typing import Optional, Iterable, Union, Literal, TYPE_CHECKING, Annotated
from abc import ABC, abstractmethod
import numpy as np
from .magnetsettings import MagnetSettings
from .rng import RNG

if TYPE_CHECKING:
    from .simulated_commissioning import SimulatedCommissioning

import logging
logger = logging.getLogger(__name__)

class KickProgram(BaseModel, ABC, extra='forbid'):
    control: str

    _generator: Optional[Iterable] = PrivateAttr(default=None)
    _buffer: list[float] = PrivateAttr(default_factory=list)
    _initial_value: float = PrivateAttr(default=0)
    _magnet_settings: Optional[MagnetSettings] = PrivateAttr(default=None)
    _rng: Optional[RNG] = PrivateAttr(default=None)

    @abstractmethod
    def get_generator(self): ...

    def initialize(self, magnet_settings: MagnetSettings):
        self._magnet_settings = magnet_settings
        self._generator = self.get_generator()
        self._buffer = []
        self._initial_value = self._magnet_settings.get(self.control)
        self._rng = magnet_settings._parent.rng

    def apply_next_turn(self):
        value = next(self._generator)
        self._buffer.append(value)
        self._magnet_settings.set(self.control, value + self._initial_value)

    def finalize(self):
        self._magnet_settings.set(self.control, self._initial_value)

class SingleKickProgram(KickProgram):
    kind: Literal["single_kick"] = "single_kick"
    turn_to_kick: int
    amplitude: float

    def get_generator(self):
        turn = 0
        while 1:
            if self.turn_to_kick == turn:
                yield self.amplitude
            else:
                yield 0.
            turn += 1

class ACProgram(KickProgram):
    kind: Literal["ac"] = "ac"
    tune: float = 0
    ramp_up_turns: int = 0
    flat_top_turns: int = 1000
    ramp_down_turns: int = 0
    amplitude: float

    def get_generator(self):
        t1 = self.ramp_up_turns
        t2 = self.ramp_up_turns + self.flat_top_turns
        t3 = self.ramp_up_turns + self.flat_top_turns + self.ramp_down_turns

        for turn in range(t1):
            yield ( ( turn / self.ramp_up_turns ) * self.amplitude ) * np.sin(2 * np.pi * self.tune * turn)

        for turn in range(t1, t2):
            yield self.amplitude * np.sin(2 * np.pi * self.tune * turn)

        for turn in range(t2, t3):
            yield ( (1 - (turn - t2) / self.ramp_down_turns ) * self.amplitude ) * np.sin(2 * np.pi * self.tune * turn)

        while 1:
            yield 0

class WhiteNoiseProgram(KickProgram):
    kind: Literal["white_noise"] = "white_noise"
    amplitude: float

    def get_generator(self):
        while 1:
            yield self._rng.normal() * self.amplitude


KickProgramType = Annotated[
    SingleKickProgram | ACProgram | WhiteNoiseProgram,
    Field(discriminator="kind"),
]

class KickerSettings(BaseModel, extra="forbid"):
    programs: dict[str, KickProgramType] = Field(default_factory=dict)
    active_programs: list[str] = Field(default_factory=list)

    _parent: Optional["SimulatedCommissioning"] = PrivateAttr(default=None)

    @property
    def has_active(self) -> bool:
        return bool(self.active_programs)

    def activate(self, name: str):
        if name not in self.programs.keys():
            raise Exception(f"Kick program {name} not found in kicker settings.")

        if name in self.active_programs:
            logger.warning(f"Kick program {name} is already active.")
        else:
            control = self.programs[name].control
            for active_name in self.active_programs:
                active_control = self.programs[active_name].control
                if active_control == control:
                    raise ValueError(
                        f"Cannot activate kick program {name}: control {control} is already "
                        f"used by active kick program {active_name}."
                    )
            self.active_programs.append(name)

    def deactivate(self, name: Optional[str] = None):
        if name is None:
            self.active_programs = []
            return

        if name not in self.programs.keys():
            raise Exception(f"Kick program {name} not found in kicker settings.")

        if name in self.active_programs:
            self.active_programs.pop(self.active_programs.index(name))
        else:
            logger.warning(f"Kick program {name} is already not active.")

    def _check_control_exists(self, name: str, control: str):
        SC = self._parent
        if control not in SC.magnet_settings.controls:
            raise Exception(f"Control {control} not found in magnet settings when adding {name} kick program.")

    def add_single_kick_program(self, name: str, control: str, amplitude: float, turn_to_kick: int = 0):
        self._check_control_exists(name, control)
        self.programs[name] = SingleKickProgram(control=control, turn_to_kick=turn_to_kick, amplitude=amplitude) 

    def add_ac_program(self, name: str, control: str, amplitude: float, tune: float = 0,
                       ramp_up_turns: float = 0, flat_top_turns: int = 1000, ramp_down_turns: int = 0):
        self._check_control_exists(name, control)
        self.programs[name] = ACProgram(control=control, amplitude=amplitude, tune=tune, ramp_up_turns=ramp_up_turns,
                                        flat_top_turns=flat_top_turns, ramp_down_turns=ramp_down_turns)

    def add_white_noise_program(self, name: str, control: str, amplitude: float):
        self._check_control_exists(name, control)
        self.programs[name] = WhiteNoiseProgram(control=control, amplitude=amplitude)

    def initialize(self, magnet_settings: MagnetSettings):
        for name in self.active_programs:
            self.programs[name].initialize(magnet_settings)

    def apply_next_turn(self):
        for name in self.active_programs:
            self.programs[name].apply_next_turn()

    def finalize(self):
        for name in self.active_programs:
            self.programs[name].finalize()
