from typing import Optional, Union, TYPE_CHECKING
from pydantic import BaseModel, PrivateAttr
import numpy as np

if TYPE_CHECKING:
    from .simulated_commissioning import SimulatedCommissioning

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
    # x_size: float = 1
    # px_divergence: float = 1
    # y_size: float = 1
    # py_divergence: float = 1
    bunch_length: float = 1
    energy_spread: float = 1
    betx: float = 1
    alfx: float = 0
    bety: float = 1
    alfy: float = 0
    gemit_x: float = 1
    gemit_y: float = 1

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

    @property
    def invW(self) -> np.ndarray:
        invW = np.zeros([6,6])

        sbetx = self.betx**0.5
        invW[0, 0] = sbetx
        invW[1, 0] = -self.alfx / sbetx
        invW[1, 1] = 1 / sbetx

        sbety = self.bety**0.5
        invW[2, 2] = sbety
        invW[3, 2] = -self.alfy / sbety
        invW[3, 3] = 1 / sbety

        invW[4,4] = 1
        invW[5,5] = 1

    def generate_orbit_centered_bunch(self, use_design=False) -> np.ndarray:
        # When array will be transposed to go to AT, it will be F_CONTIGUOUS :)
        bunch = np.zeros([self.n_particles, 6])
        if self.n_particles > 1:
            bunch_norm = np.zeros([self.n_particles, 6])
            bunch_norm[:, 0] = self._parent.rng.normal(size=self.n_particles)
            bunch_norm[:, 1] = self._parent.rng.normal(size=self.n_particles)
            bunch_norm[:, 2] = self._parent.rng.normal(size=self.n_particles)
            bunch_norm[:, 3] = self._parent.rng.normal(size=self.n_particles)
            bunch_norm[:, 4] = self._parent.rng.normal(size=self.n_particles)
            bunch_norm[:, 5] = self._parent.rng.normal(size=self.n_particles)
            raise NotImplementedError
        
        twiss = self._parent.lattice.get_twiss(use_design=use_design)

        bunch[:, 0] += twiss['x'][0]
        bunch[:, 1] += twiss['px'][0]
        bunch[:, 2] += twiss['y'][0]
        bunch[:, 3] += twiss['py'][0]
        bunch[:, 4] += twiss['tau'][0]
        bunch[:, 5] += twiss['delta'][0]
        return bunch

    def generate_zero_centered_bunch(self) -> np.ndarray:
        # When array will be transposed to go to AT, it will be F_CONTIGUOUS :)
        bunch = np.zeros([self.n_particles, 6])
        if self.n_particles > 1:
            bunch_norm = np.zeros([self.n_particles, 6])
            bunch_norm[:, 0] = self._parent.rng.normal(size=self.n_particles)
            bunch_norm[:, 1] = self._parent.rng.normal(size=self.n_particles)
            bunch_norm[:, 2] = self._parent.rng.normal(size=self.n_particles)
            bunch_norm[:, 3] = self._parent.rng.normal(size=self.n_particles)
            bunch_norm[:, 4] = self._parent.rng.normal(size=self.n_particles)
            bunch_norm[:, 5] = self._parent.rng.normal(size=self.n_particles)

            sbetx = self.betx**0.5
            bunch[:, 0] = sbetx * bunch_norm[:, 0]
            bunch[:, 1] = -self.alfx / sbetx * bunch_norm[:, 0] + ( 1. / sbetx ) * bunch_norm[: , 1]

            sbety = self.bety**0.5
            bunch[:, 2] = sbety * bunch_norm[:, 2]
            bunch[:, 3] = -self.alfx / sbety * bunch_norm[:, 2] + ( 1. / sbety ) * bunch_norm[: , 3]

            bunch[:, 4] = bunch_norm[:, 4]
            bunch[:, 5] = bunch_norm[:, 5]

            sgemit_x = self.gemit_x**0.5
            bunch[:, 0] = sgemit_x * bunch[:, 0] 
            bunch[:, 1] = sgemit_x * bunch[:, 1] 
            sgemit_y = self.gemit_y**0.5
            bunch[:, 2] = sgemit_y * bunch[:, 2] 
            bunch[:, 3] = sgemit_y * bunch[:, 3] 
            bunch[:, 4] = self.bunch_length * bunch[:, 4]
            bunch[:, 5] = self.energy_spread * bunch[:, 5]
        return bunch

    def generate_bunch(self, use_design=False) -> np.ndarray:
        # When array will be transposed to go to AT, it will be F_CONTIGUOUS :)
        bunch = self.generate_zero_centered_bunch()
        if use_design:
            bunch[:, 0] += self.x
            bunch[:, 1] += self.px
            bunch[:, 2] += self.y
            bunch[:, 3] += self.py
            bunch[:, 4] += self.tau
            bunch[:, 5] += self.delta
        else:
            bunch[:, 0] += self.x_inj
            bunch[:, 1] += self.px_inj
            bunch[:, 2] += self.y_inj
            bunch[:, 3] += self.py_inj
            bunch[:, 4] += self.tau_inj
            bunch[:, 5] += self.delta_inj
        return bunch
