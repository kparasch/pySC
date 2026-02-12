from __future__ import annotations
from typing import Dict, Optional, Union, TYPE_CHECKING
from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from .simulated_commissioning import SimulatedCommissioning

CAVITY_NAME_TYPE = Union[str, int]

class RFCavity(BaseModel, extra="forbid"):
    sim_index: int

    # unintentional error (w.r.t. setpoints of the parent system)
    voltage_error: float = 0
    phase_error: float = 0
    frequency_error: float = 0

    # correction factors to be found during commissioning,
    # if found correctly, they will be opposite to the above errors.
    voltage_correction: float = 0
    phase_correction: float = 0
    frequency_correction: float = 0

    # intentional setpoints
    voltage_delta: float = 0
    phase_delta: float = 0
    frequency_delta: float = 0

    to_design: bool = False
    _parent_system: Optional[RFSystem] = PrivateAttr(default = None)

    @property
    def actual_voltage(self):
        system = self._parent_system
        total_voltage = system.voltage
        cavity_voltage = total_voltage / len(system.cavities)
        return (cavity_voltage + self.voltage_error 
                + self.voltage_correction + self.voltage_delta)
    @property
    def actual_phase(self):
        system = self._parent_system
        return (system.phase + self.phase_error 
                + self.phase_correction + self.phase_delta)

    @property
    def actual_frequency(self):
        system = self._parent_system
        return (system.frequency + self.frequency_error 
                + self.frequency_correction + self.frequency_delta)

    def update(self) -> None:
        SC = self._parent_system._parent._parent
        voltage = self.actual_voltage
        phase = self.actual_phase
        frequency = self.actual_frequency
        SC.lattice.update_cavity(index=self.sim_index, voltage=voltage, phase=phase,
                              frequency=frequency, use_design=self.to_design)
        return

class RFSystem(BaseModel, extra="forbid"):
    cavities: list[CAVITY_NAME_TYPE] = []
    voltage: float # V
    phase: float # degrees
    frequency: float # Hz
    _parent: RFSettings = PrivateAttr(default = 0)

    def set_voltage(self, voltage: float):
        self.voltage = voltage
        self.trigger_update()

    def set_phase(self, phase: float):
        self.phase = phase
        self.trigger_update()

    def set_frequency(self, frequency: float):
        self.frequency = frequency
        self.trigger_update()

    def trigger_update(self):
        for cav in self.cavities:
            self._parent.cavities[cav].update()

    @property
    def indices(self) -> list[int]:
        return [self._parent.cavities[cav].sim_index for cav in self.cavities]


class RFSettings(BaseModel, extra="forbid"):
    systems: Dict[str, RFSystem] = Field(default_factory=dict)
    cavities: Dict[CAVITY_NAME_TYPE, RFCavity] = Field(default_factory=dict)
    _parent: Optional["SimulatedCommissioning"] = PrivateAttr(default=None)

    @property
    def main(self):
        return self.systems['main']