from abc import ABC
from pydantic import BaseModel, Field
import numpy as np

from ..core.new_simulated_commissioning import SimulatedCommissioning

def function_is_overriden(func):
    obj = func.__self__
    abs_func = getattr(super(type(obj), obj), func.__name__)
    return func.__func__ != abs_func.__func__

class AbstractInterface(BaseModel, ABC):
    def get_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        returns the orbit of the machine in two lists/arrays:
        e.g. x, y = interface.get_orbit()

        we should make sure here that we can call this function twice and not get the same reading (i.e. wait that the orbit has refreshed).
        '''
        raise NotImplementedError

    def get(self, name: str) -> float:
        '''
        gets a magnet strength in the machine (physics units), integrated or not.
        e.g. k0 = interface.get('SD1A-C01-H')
        '''
        raise NotImplementedError

    def set(self, name: str, value: float):
        '''
        sets a magnet strength in the machine (physics units), integrated or not.
        e.g. interface.set('SD1A-C01-H', 100e-6)

        waiting time to make sure power supply is settled and eddy currents are decayed to be handled also here.
        '''
        raise NotImplementedError

    def get_many(self, names: list) -> dict[str, float]:
        '''
        gets many magnet strengths in the machine (physics units), integrated or not.
        e.g. data_k0 = interface.get_many(['SD1A-C01-H', 'SD1A-C01-V'])
        resulting in data_k0 as {'SD1A-C01-H': 100e-6, 'SD1A-C01-V': 200e-6} for example
        '''
        raise NotImplementedError

    def set_many(self, data: dict[str, float]):
        '''
        perhaps there is a specific host you use to set all corrector settings?

        sets many magnet strengths in the machine (physics units), integrated or not.
        e.g. interface.set_many({'SD1A-C01-H': 100e-6, 'SD1A-C01-V': 200e-6})

        waiting time to make sure power supply is settled and eddy currents are decayed to be handled also here.
        '''
        raise NotImplementedError

class pySCOrbitInterface(AbstractInterface):
    SC: SimulatedCommissioning = Field(repr=False)

    def get_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        return self.SC.bpm_system.capture_orbit()

    def get_ref_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        return self.SC.bpm_system.reference_x, self.SC.bpm_system.reference_y

    def get(self, name: str) -> float:
        return self.SC.magnet_settings.get(name)

    def set(self, name: str, value: float):
        self.SC.magnet_settings.set(name, value)
        return

    def get_many(self, names: list) -> dict[str, float]:
        return self.SC.magnet_settings.get_many(names)

    def set_many(self, data: dict[str, float]):
        self.SC.magnet_settings.set_many(data)
        return

    def get_rf_main_frequency(self) -> float:
        return self.SC.rf_settings.main.frequency

    def set_rf_main_frequency(self, frequency: float):
        self.SC.rf_settings.main.set_frequency(frequency)
        return

class pySCInjectionInterface(pySCOrbitInterface):
    SC: SimulatedCommissioning = Field(repr=False)
    n_turns: int = 1

    def get_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        x,y= self.SC.bpm_system.capture_injection(n_turns=self.n_turns)
        return x.flatten(order='F'), y.flatten(order='F')

    def get_ref_orbit(self) -> tuple[np.ndarray, np.ndarray]:
        x_ref = np.repeat(self.SC.bpm_system.reference_x[:, np.newaxis], self.n_turns, axis=1)
        y_ref = np.repeat(self.SC.bpm_system.reference_y[:, np.newaxis], self.n_turns, axis=1)
        return x_ref.flatten(order='F'), y_ref.flatten(order='F')
