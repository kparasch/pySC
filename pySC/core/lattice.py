from pydantic import BaseModel, PrivateAttr, model_validator
from typing import Optional, Literal
import re
import numpy as np
import at
from scipy.constants import c as C_LIGHT
from numpy import array as nparray

import logging
logger = logging.getLogger(__name__)

class Lattice(BaseModel, extra="forbid"):
    """
    Base class for machine
    """
    lattice_file: str
    no_6d : bool = False
    naming : Optional[str] = None
    _omp_num_threads = PrivateAttr(default=None)
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
    Represents a lattice defined with AT (Accelerator Toolbox).
    """

    # fake field so that pydantic can distinguish
    # the different machine types
    at_simulator: None = None
    use: str = 'RING'
    orbit_guess: Optional[list[float]] = None
    use_orbit_guess: bool = False
    _ring: at.Lattice = PrivateAttr(default=None)
    _design: at.Lattice = PrivateAttr(default=None)

    @model_validator(mode="after")
    def load_lattice(self):
        self._ring = at.load_mat(self.lattice_file, use=self.use)
        self._design = at.load_mat(self.lattice_file, use=self.use)

        if not self.no_6d:
            self._ring.enable_6d()
            self._design.enable_6d()

        if self.orbit_guess is None:
            self.orbit_guess = [0] * 6

        self._twiss = self.get_twiss(use_design=True)

        return self

    @property
    def omp_num_threads(self):
        return self._omp_num_threads

    @omp_num_threads.setter
    def omp_num_threads(self, value: int):
        self._omp_num_threads = value
        at.lattice.DConstant.patpass_poolsize = value

    def get_name_from_index(self, index: int):
        if self.naming is None:
            return str(index)
        else:
            if hasattr(self.design[index], self.naming):
                return getattr(self.design[index], self.naming)
            else:
                logger.warning(f'{self.design[index].FamName} at index {index} has no "{self.naming}" field. Falling back to index-based name.')
                return str(index)

    def update_orbit_guess(self, n_turns=1000):
        bunch = np.zeros([1,6])
        out = self.track(bunch, n_turns=n_turns, coordinates=['x','px','y','py','delta','tau'])
        guess = np.nanmean(out, axis=3).flatten()
        logger.info('Found orbit guess:')
        logger.info(f'  x = {guess[0]}')
        logger.info(f'  px = {guess[1]}')
        logger.info(f'  y = {guess[2]}')
        logger.info(f'  py = {guess[3]}')
        logger.info(f'  tau = {guess[5]}')
        logger.info(f'  delta = {guess[4]}')
        self.orbit_guess = list(guess)

    def track(self, bunch: nparray, indices: Optional[list[int]] = None, n_turns: int = 1, use_design: bool = False, coordinates: Optional[list] = None) -> nparray:
        new_bunch = bunch.copy()
        new_bunch[:,4], new_bunch[:,5] = new_bunch[:,5].copy(), new_bunch[:,4].copy()  # swap zeta and delta for AT
        if use_design:
            ring = self._design
        else:
            ring = self._ring

        ## transform coordinates to indices
        if coordinates is None:
            coordinates = ['x', 'y']
        coord_map = {'x':0, 'px':1, 'y':2, 'py':3, 'tau':5, 'delta':4}
        coords = [coord_map[c] for c in coordinates]

        if indices is not None:
            if self.omp_num_threads is not None:
                out = at.patpass(ring, new_bunch.T, refpts=indices, nturns=n_turns)
            else:
                out = ring.track(new_bunch.T, refpts=indices, nturns=n_turns)[0]
            #out = self._design.track(bunch.T, refpts=indices, nturns=n_turns)[0]
        else:
            if self.omp_num_threads is not None:
                out = at.patpass(ring, new_bunch.T, nturns=n_turns)
            else:
                out = ring.track(new_bunch.T, nturns=n_turns)[0]
            # out = ring.track(bunch.T, nturns=n_turns)[0]
        #     if indices is not None:
        #         out = ring.track(bunch.T, refpts=indices, nturns=n_turns)[0]
        #     else:
        #         out = ring.track(bunch.T, nturns=n_turns)[0]
        xy = out[coords, :, :, :]
        return xy

    def get_orbit(self, indices: list[int] = None, use_design=False) -> dict:
        """
        Returns the closed orbit for the specified indices.
        If no indices are provided, returns the closed orbit for all elements.
        """
        if indices is None:
            indices = range(len(self._design))
        ring = self._design if use_design else self._ring
        if self.use_orbit_guess:
            assert self.no_6d is False, "Using orbit guesses with a 4D lattice is not checked/implemented."
            _, orbit = at.find_orbit(ring, refpts=indices, guess=np.array(self.orbit_guess))
        else:
            _, orbit = at.find_orbit(ring, refpts=indices)

        return orbit[:, [0,2]].T

    def get_twiss(self, indices: Optional[list[int]] = None, use_design=False) -> dict:
        """
        Returns the twiss parameters for the specified indices.
        If no indices are provided, returns all twiss parameters.
        """
        if indices is None:
            indices = range(len(self._design))
        ring = self._design if use_design else self._ring
        if self.use_orbit_guess:
            assert self.no_6d is False, "Using orbit guesses with a 4D lattice is not checked/implemented."
            orbit0, _ = at.find_orbit(ring, refpts=indices, guess=np.array(self.orbit_guess))
        else:
            orbit0, _ = at.find_orbit(ring, refpts=indices)

        _, ringdata, elemdata = at.get_optics(ring, refpts=indices, get_chrom=True, orbit=orbit0)

        qs = ringdata['tune'][2] if not self.no_6d else 0 # doesn't exist when ring has 6d disabled

        twiss = {'qx': elemdata.mu[-1,0]/2/np.pi,
                 'qy': elemdata.mu[-1,1]/2/np.pi,
                 'qs': qs,
                 'dqx': ringdata['chromaticity'][0],
                 'dqy': ringdata['chromaticity'][1],
                 's': elemdata.s_pos,
                 'x' : elemdata.closed_orbit[:, 0],
                 'px': elemdata.closed_orbit[:, 1],
                 'y' : elemdata.closed_orbit[:, 2],
                 'py': elemdata.closed_orbit[:, 3],
                 'delta': elemdata.closed_orbit[:, 4],
                 'tau': elemdata.closed_orbit[:, 5],
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

    def get_tune(self, method='6d', use_design=False) -> tuple[float, float]:
        assert method in ['4d', '6d']
        ring = self._design if use_design else self._ring

        if self.no_6d:
            logger.warning("Lattice has 6d disabled, using 4d method instead.")
            method = '4d'

        if method == '4d' and not self.no_6d:
            ring.disable_6d()

        try:
            tunes = ring.get_tune()
            qx = tunes[0]
            qy = tunes[1]
        except Exception as e:
            logger.error(f"Error computing tune, {type(e)}: {e}")
            qx = np.nan
            qy = np.nan

        if method == '4d' and not self.no_6d:
            ring.enable_6d()

        return qx, qy

    def get_chromaticity(self, method='6d', use_design=False) -> tuple[float, float]:
        assert method in ['4d', '6d']
        ring = self._design if use_design else self._ring

        if self.no_6d:
            logger.warning("Lattice has 6d disabled, using 4d method instead.")
            method = '4d'

        if method == '4d' and not self.no_6d:
            ring.disable_6d()

        try:
            chroms = ring.get_chrom()
            dqx = chroms[0]
            dqy = chroms[1]
        except Exception as e:
            logger.error(f"Error computing chromaticity, {type(e)}: {e}")
            dqx = np.nan
            dqy = np.nan

        if method == '4d' and not self.no_6d:
            ring.enable_6d()

        return dqx, dqy

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
        if self._design[index].Length:
            return self._design[index].Length
        else: # when length is zero
            return 1

    def get_magnet_component(self, index: int, component_type: Literal['A', 'B'],
                             order: int, use_design=True) -> float:
        assert component_type in ['A', 'B']
        if use_design:
            elem = self._design[index]
        else:
            elem = self._ring[index]

        if type(elem) is at.Corrector:
            if order != 0:
                raise Exception(f'ERROR: order={order}, for at.Corrector, order different than 0 is not supported.')
            length = self.get_length(index)
            if component_type == 'A':
                value = elem.KickAngle[1] / length
            else:
                value = - elem.KickAngle[0] / length
        else:
            if hasattr(elem, 'KickAngle') and (elem.KickAngle[0] != 0 or elem.KickAngle[1] != 0):
                logger.warning('Non-corrector element has non-zero KickAngle. Be careful. Ask for help.')

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
            length = self.get_length(index)
            if component_type == 'A':
                elem.KickAngle[1] = value * length
            elif component_type == 'B':
                elem.KickAngle[0] = - value * length
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

    def one_turn_matrix(self, use_design=False):
        ring = self._design if use_design else self._ring

        if self.use_orbit_guess:
            assert self.no_6d is False, "Using orbit guesses with a 4D lattice is not checked/implemented."
            orbit0, _ = at.find_orbit(ring, guess=np.array(self.orbit_guess))
        else:
            orbit0, _ = at.find_orbit(ring)

        if self.no_6d:
            M = ring.find_m44(orbit=orbit0)[0]
        else:
            M = ring.find_m66(orbit=orbit0)[0]

        return M
