from pydantic import Field
import numpy as np
from typing import TYPE_CHECKING

from ..apps.interface import AbstractInterface
if TYPE_CHECKING:
    from ..core.simulated_commissioning import SimulatedCommissioning

class pySCOrbitInterface(AbstractInterface):
    SC: "SimulatedCommissioning" = Field(repr=False)
    use_design: bool = False
    bba: bool = True
    subtract_reference: bool = True

    def get_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        return self.SC.bpm_system.capture_orbit(use_design=self.use_design, bba=self.bba,
                                                subtract_reference=self.subtract_reference)

    def get_ref_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        return self.SC.bpm_system.reference_x, self.SC.bpm_system.reference_y

    def get(self, name: str) -> float:
        return self.SC.magnet_settings.get(name, use_design=self.use_design)

    def set(self, name: str, value: float):
        self.SC.magnet_settings.set(name, value, use_design=self.use_design)
        return

    def get_many(self, names: list) -> dict[str, float]:
        return self.SC.magnet_settings.get_many(names, use_design=self.use_design)

    def set_many(self, data: dict[str, float]):
        self.SC.magnet_settings.set_many(data, use_design=self.use_design)
        return

    def get_rf_main_frequency(self) -> float:
        if self.use_design:
            rf_settings = self.SC.design_rf_settings
        else:
            rf_settings = self.SC.rf_settings

        return rf_settings.main.frequency

    def set_rf_main_frequency(self, frequency: float):
        if self.use_design:
            rf_settings = self.SC.design_rf_settings
        else:
            rf_settings = self.SC.rf_settings

        rf_settings.main.set_frequency(frequency)
        return

class pySCInjectionInterface(pySCOrbitInterface):
    SC: "SimulatedCommissioning" = Field(repr=False)
    n_turns: int = 1

    def get_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        x,y= self.SC.bpm_system.capture_injection(n_turns=self.n_turns, use_design=self.use_design,
                                                  bba=self.bba, subtract_reference=self.subtract_reference)
        return x.flatten(order='F'), y.flatten(order='F')

    def get_ref_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        x_ref = np.repeat(self.SC.bpm_system.reference_x[:, np.newaxis], self.n_turns, axis=1)
        y_ref = np.repeat(self.SC.bpm_system.reference_y[:, np.newaxis], self.n_turns, axis=1)
        return x_ref.flatten(order='F'), y_ref.flatten(order='F')
