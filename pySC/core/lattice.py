from pydantic import BaseModel, PrivateAttr, model_validator
from typing import Optional, Literal
import re
import at
from scipy.constants import c as C_LIGHT
from numpy import array as nparray

class Lattice(BaseModel, extra="forbid"):
    """
    Base class for machine
    """
    lattice_file: str
    _ring = PrivateAttr(default=None)
    _design = PrivateAttr(default=None)
    _twiss: dict = PrivateAttr(default=None)

    @property
    def ring(self):
        return self._ring

    @property
    def design(self):
        return self._design

    @property
    def twiss(self):
        return self._twiss


class XSuiteLattice(Lattice):
    """
    Not implemented.
    """

    # fake field so that pydantic can distinguish
    # the different machine types
    xsuite_simulator: None = None

class ATLattice(Lattice):
    """
    Represents a machine in the AA (Accelerator Analysis) framework.
    This class is used to define the properties and behaviors of a machine.
    """

    # fake field so that pydantic can distinguish
    # the different machine types
    at_simulator: None = None
    _ring: at.Lattice = PrivateAttr(default=None)
    _design: at.Lattice = PrivateAttr(default=None)

    @model_validator(mode="after")
    def load_lattice(self):
        self._ring = at.load_mat(self.lattice_file)
        self._design = at.load_mat(self.lattice_file)

        self._ring.enable_6d()
        self._design.enable_6d()

        self._twiss = self.get_twiss(use_design=True)
        return self

    def get_orbit(self, indices: list[int] = None, use_design=False) -> dict:
        """
        Returns the closed orbit for the specified indices.
        If no indices are provided, returns the closed orbit for all elements.
        """
        if indices is None:
            indices = range(len(self._design))
        ring = self._design if use_design else self._ring
        _, orbit = at.find_orbit6(ring, refpts=indices)
        return orbit[:, [0,2]].T

    def get_twiss(self, indices: Optional[list[int]] = None, use_design=False) -> dict:
        """
        Returns the twiss parameters for the specified indices.
        If no indices are provided, returns all twiss parameters.
        """
        if indices is None:
            indices = range(len(self._design))
        ring = self._design if use_design else self._ring
        _, ringdata, elemdata = at.get_optics(ring, refpts=indices, get_chrom=True)
        twiss = {'qx': ringdata['tune'][0],
                 'qy': ringdata['tune'][1],
                 'qs': ringdata['tune'][2],
                 'dqx': ringdata['chromaticity'][0],
                 'dqy': ringdata['chromaticity'][1],
                 's': elemdata.s_pos,
                 'x' : elemdata.closed_orbit[:, 0],
                 'px': elemdata.closed_orbit[:, 1],
                 'y' : elemdata.closed_orbit[:, 2],
                 'py': elemdata.closed_orbit[:, 3],
                 'delta': elemdata.closed_orbit[:, 4],
                 'zeta': elemdata.closed_orbit[:, 5],
                 'betx': elemdata.beta[:, 0],
                 'bety': elemdata.beta[:, 1],
                 'alfx': elemdata.alpha[:, 0],
                 'alfy': elemdata.alpha[:, 1],
                 'mux': elemdata.mu[:, 0],
                 'muy': elemdata.mu[:, 1],
                 'dx' : elemdata.dispersion[:, 0],
                 'dpx': elemdata.dispersion[:, 1],
                 'dy' : elemdata.dispersion[:, 2],
                 'dpy': elemdata.dispersion[:, 3],
                }
        return twiss

    # def update_magnet(self, index: int, A : list, B : list, max_order : int) -> None:
    #     if max_order + 1 != len(A) or max_order + 1 != len(B):
    #         raise ValueError(f"Length of A and B must be {max_order + 1} for index {index}.")

    #     if type(self._ring[index]) is at.lattice.elements.Corrector:
    #         kickangle = np.array([-B[0], A[0]]) * self._ring[index].Length
    #         self._ring[index].KickAngle = kickangle
    #     else:
    #         self._ring[index].PolynomA = np.array(A, dtype=float)
    #         self._ring[index].PolynomB = np.array(B, dtype=float)
    #         self._ring[index].MaxOrder = max_order
    #     return

    def find_with_regex(self, regex: str) -> list[int]:
        """
        Find elements in the ring that match the given regular expression.
        Returns a list of indices of the matching elements.
        """
        indices = [ind for ind, el in enumerate(self._ring) if re.search(regex, el.FamName)]
        return indices

    def is_dipole(self, index: int) -> bool:
        """
        Check if the element at the given index is a dipole with bending angle.
        """
        elem = self._design[index]
        if hasattr(elem, 'BendingAngle') and elem.BendingAngle != 0:
            return True
        else:
            return False

    def get_bending_angle(self, index: int) -> float:
        elem  = self._design[index]
        return getattr(elem, 'BendingAngle', 0)

    def get_length(self, index: int) -> float:
        return self._design[index].Length

    def get_magnet_component(self, index: int, component_type: Literal['A', 'B'],
                             order: int, use_design=True) -> float:
        assert component_type in ['A', 'B']
        if use_design:
            elem = self._design[index]
        else:
            elem = self._ring[index]

        if type(elem) is at.Corrector:
            if order != 0:
                raise Exception('ERROR: order={order}, for at.Corrector, order different than 0 is not supported.')
            if component_type == 'A':
                value = elem.KickAngle[1] / elem.Length
            else:
                value = - elem.KickAngle[0] / elem.Length
        else:
            value =  getattr(elem, f'Polynom{component_type}')[order]

        return value

    def set_magnet_component(self, index: int, value: float,
                             component_type: Literal['A', 'B'],
                             order: int, use_design=True) -> None:
        assert component_type in ['A', 'B']
        if use_design:
            elem = self._design[index]
        else:
            elem = self._ring[index]

        if type(elem) is at.Corrector:
            if order != 0:
                raise Exception('ERROR: order={order}, for at.Corrector, order different than 0 is not supported.')
            if component_type == 'A':
                elem.KickAngle[1] = value * elem.Length
            elif component_type == 'B':
                elem.KickAngle[0] = - value * elem.Length
            else:
                raise Exception('ERROR: Not supposed to happen!')
        else:
            if component_type == 'A':
                elem.PolynomA[order] = value
            elif component_type == 'B':
                elem.PolynomB[order] = value
            else:
                raise Exception('ERROR: Not supposed to happen!')
        return

    def get_cavity_voltage_phase_frequency(self, index: int, use_design=True):
        if use_design:
            elem = self._design[index]
        else:
            elem = self._ring[index]

        assert type(elem) is at.RFCavity

        voltage = elem.Voltage
        timelag = elem.TimeLag
        frequency = elem.Frequency

        phase = timelag * (360 * frequency) / C_LIGHT
        phase = phase % 360

        return voltage, phase, frequency

    def update_cavity(self, index: int, voltage: float, phase: float, frequency: float, use_design=True):
        if use_design:
            elem = self._design[index]
        else:
            elem = self._ring[index]

        assert type(elem) is at.RFCavity

        timelag = phase * C_LIGHT / (360. * frequency)
        elem.Voltage = voltage
        elem.TimeLag = timelag
        elem.Frequency = frequency
        return

    def track(self, bunch: nparray, indices: Optional[list[int]] = None, n_turns: int = 1, use_design: bool = False) -> tuple[nparray, nparray]:
        if use_design:
            if indices is not None:
                out = self._design.track(bunch.T, refpts=indices, nturns=n_turns)[0]
            else:
                out = self._design.track(bunch.T, nturns=n_turns)[0]
        else:
            if indices is not None:
                out = self._ring.track(bunch.T, refpts=indices, nturns=n_turns)[0]
            else:
                out = self._ring.track(bunch.T, nturns=n_turns)[0]
        xy = out[[0,2], :, :, :]
        return xy
