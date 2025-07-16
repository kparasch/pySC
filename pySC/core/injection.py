from typing import Optional, Union, TYPE_CHECKING
from pydantic import BaseModel, PrivateAttr
import numpy as np

if TYPE_CHECKING:
    from .new_simulated_commissioning import SimulatedCommissioning

CAVITY_NAME_TYPE = Union[str, int]

class InjectionSettings(BaseModel, extra="forbid"):
    x: float = 0
    px: float = 0
    y: float = 0
    py: float = 0
    tau: float = 0
    delta: float = 0

    x_error_syst: float = 0
    px_error_syst: float = 0
    y_error_syst: float = 0
    py_error_syst: float = 0
    tau_error_syst: float = 0
    delta_error_syst: float = 0

    x_error_stat: float = 0
    px_error_stat: float = 0
    y_error_stat: float = 0
    py_error_stat: float = 0
    tau_error_stat: float = 0
    delta_error_stat: float = 0

    n_particles: int = 1
    x_size: float = 1
    px_divergence: float = 1
    y_size: float = 1
    py_divergence: float = 1
    bunch_length: float = 1
    energy_spread: float = 1

    _parent: Optional["SimulatedCommissioning"] = PrivateAttr(default=None)

    @property
    def x_inj(self):
        return self.x + self._parent.rng.normal(loc=self.x_error_syst, scale=self.x_error_stat) 

    @property
    def px_inj(self):
        return self.px + self._parent.rng.normal(loc=self.px_error_syst, scale=self.px_error_stat) 

    @property
    def y_inj(self):
        return self.y + self._parent.rng.normal(loc=self.y_error_syst, scale=self.y_error_stat) 

    @property
    def py_inj(self):
        return self.py + self._parent.rng.normal(loc=self.py_error_syst, scale=self.py_error_stat) 

    @property
    def tau_inj(self):
        return self.tau + self._parent.rng.normal(loc=self.tau_error_syst, scale=self.tau_error_stat) 

    @property
    def delta_inj(self) -> float:
        return self.delta + self._parent.rng.normal(loc=self.delta_error_syst, scale=self.delta_error_stat) 

    def generate_bunch(self) -> np.ndarray:
        # When array will be transposed to go to AT, it will be F_CONTIGUOUS :)
        bunch = np.zeros([self.n_particles, 6])

        if self.n_particles == 1:
            bunch[0, 0] = self.x_inj
            bunch[0, 1] = self.px_inj
            bunch[0, 2] = self.y_inj
            bunch[0, 3] = self.py_inj
            bunch[0, 4] = self.tau_inj
            bunch[0, 5] = self.delta_inj
        else:
            bunch[:, 0] = self._parent.rng.normal(loc=self.x_inj, scale=self.x_size, size=self.n_particles)
            bunch[:, 1] = self._parent.rng.normal(loc=self.px_inj, scale=self.px_divergence, size=self.n_particles)
            bunch[:, 2] = self._parent.rng.normal(loc=self.y_inj, scale=self.y_size, size=self.n_particles)
            bunch[:, 3] = self._parent.rng.normal(loc=self.py_inj, scale=self.py_divergence, size=self.n_particles)
            bunch[:, 4] = self._parent.rng.normal(loc=self.tau_inj, scale=self.bunch_length, size=self.n_particles)
            bunch[:, 5] = self._parent.rng.normal(loc=self.delta_inj, scale=self.energy_spread, size=self.n_particles)
        return bunch

