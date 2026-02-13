from abc import ABC
from pydantic import BaseModel
import numpy as np


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
