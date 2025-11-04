from pydantic import BaseModel, PrivateAttr
from typing import Optional
import datetime
import logging
import numpy as np
from pathlib import Path

from .codes import BBACode
from ..utils.file_tools import dict_to_h5
from ..tuning.tools import get_average_orbit
from .interface import AbstractInterface

from ..tuning.orbit_bba import reject_bpm_outlier, reject_center_outlier, reject_slopes, get_slopes_center, get_offset

logger = logging.getLogger(__name__)

class BBAData(BaseModel):
    """
    A class to hold the data collected during a BBA measurement.
    It contains the BPM positions and the orbit data.
    """
    quadrupole: str
    bpm: str
    corrector: str 
    plane: str
    dk0l: float  # Corrector k0 (max) step
    dk1l: float  # Quadrupole k1 step
    n0: int  # Number of steps in the corrector strength
    shots_per_orbit: int
    bipolar: bool = True
    skew_quad: bool = False
    bpm_number: int

    initial_k0l: Optional[float] = None
    initial_k1: Optional[float] = None
    timestamp: Optional[float] = None
    original_save_path: Optional[str] = None

    raw_bpm_x_center: list[list[float]] = []
    raw_bpm_y_center: list[list[float]] = []
    raw_bpm_x_up: list[list[float]] = []
    raw_bpm_y_up: list[list[float]] = []
    raw_bpm_x_down: list[list[float]] = []
    raw_bpm_y_down: list[list[float]] = []

    raw_bpm_x_center_err: list[list[float]] = []
    raw_bpm_y_center_err: list[list[float]] = []
    raw_bpm_x_up_err: list[list[float]] = []
    raw_bpm_y_up_err: list[list[float]] = []
    raw_bpm_x_down_err: list[list[float]] = []
    raw_bpm_y_down_err: list[list[float]] = []

    def save(self, folder_to_save: Optional[Path] = None) -> Path:
        if folder_to_save is None:
            folder_to_save = Path('data')
        time_str = datetime.datetime.fromtimestamp(self.timestamp).strftime("%Y%m%d_%H%M%S")
        filename = Path(folder_to_save) / Path(f'BBA_{self.bpm}_{self.plane}_{time_str}.h5')
        self.original_save_path = str(filename.resolve())
        dict_to_save = self.model_dump()
        dict_to_h5(dict_to_save, filename)
        logger.info(f'Saved data to {filename} .')
        return filename

def hysteresis_loop(name, settings, delta, n_cycles=1, bipolar=True):
    sp0 = settings.get(name)

    logger.debug(f'Hysteresis loop started for {name}.')

    for _ in range(n_cycles):
        logger.debug('    Going up to (sp0 + delta)')
        settings.set(name, sp0 + delta)
        yield BBACode.HYSTERESIS
        if bipolar:
            logger.debug('    Going down to (sp0 - delta)')
            settings.set(name, sp0 - delta)
        else:
            logger.debug('    Going down to (sp0)')
            settings.set(name, sp0)
        yield BBACode.HYSTERESIS

    if bipolar:
        logger.debug('    Going back to (sp0 - delta)')
        settings.set(name, sp0)
    yield BBACode.HYSTERESIS_DONE

class BBA_Measurement(BaseModel):
    """
    A class to perform a beam-based alignment measurement to find the quadrupole center in a storage ring.
    It sets the quadrupole and corrector strengths, collects orbit data.
    """
    bpm : str
    quadrupole: str
    h_corrector: Optional[str]
    v_corrector: Optional[str]
    dk0l_x: float
    dk1l_x: float
    dk0l_y: float
    dk1l_y: float
    n0: int = 7  # Number of steps in the corrector strength
    bpm_number: int
    shots_per_orbit: int = 2
    bipolar: bool = True
    quad_is_skew: bool = False

    initial_h_k0l: Optional[float] = None
    initial_v_k0l: Optional[float] = None
    initial_k1l: Optional[float] = None

    H_data: Optional[BBAData] = None
    V_data: Optional[BBAData] = None

    _interface: Optional[AbstractInterface] = PrivateAttr(default=None) # to be set at generation of measurement

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize BBAData instances for horizontal and vertical procedures
        if self.h_corrector is not None:
            self.H_data = BBAData(plane='X', bpm=self.bpm, quadrupole=self.quadrupole, bpm_number=self.bpm_number,
                                  corrector=self.h_corrector, dk0l=self.dk0l_x, dk1l=self.dk1l_x,
                                  n0=self.n0, shots_per_orbit=self.shots_per_orbit, bipolar=self.bipolar,
                                  skew_quad=self.quad_is_skew)
        if self.v_corrector is not None:
            self.V_data = BBAData(plane='Y', bpm=self.bpm, quadrupole=self.quadrupole, bpm_number=self.bpm_number,
                                  corrector=self.v_corrector, dk0l=self.dk0l_y, dk1l=self.dk1l_y,
                                  n0=self.n0, shots_per_orbit=self.shots_per_orbit, bipolar=self.bipolar,
                                  skew_quad=self.quad_is_skew)


    def print_init(self):
        logger.debug("Measurement plan:")
        logger.debug(f"    BPM = {self.bpm}")
        logger.debug(f"    Quadrupole = {self.quadrupole}")
        logger.debug(f"    H. corrector = {self.h_corrector}")
        logger.debug(f"        dk0l = {self.dk0l_x*1e6:.1f} urad")
        logger.debug(f"        dk1l = {self.dk1l_x:.3f} 1/m")
        logger.debug(f"    V. corrector = {self.v_corrector}")
        logger.debug(f"        dk0l = {self.dk0l_y*1e6:.1f} urad")
        logger.debug(f"        dk1l = {self.dk1l_y:.3f} 1/m")
        logger.debug(f"    number of points = {self.n0}")
        logger.debug(f"    Initial H. k0l = {self.initial_h_k0l*1e6:.1f} urad")
        logger.debug(f"    Initial V. k0l = {self.initial_v_k0l*1e6:.1f} urad")
        logger.debug(f"    Initial k1l = {self.initial_k1l:.3f} 1/m")
        logger.debug(f"    Bipolar = {self.bipolar}")
        logger.debug("")

    def one_plane_loop(self, plane: str):
        assert self._interface is not None
        assert plane in ['H', 'V']

        interface = self._interface
        if plane == 'H':
            logger.debug('Starting measurement in horizontal plane.')
            code = BBACode.HORIZONTAL
            code_done = BBACode.HORIZONTAL_DONE
        else:
            logger.debug('Starting measurement in vertical plane.')
            code = BBACode.VERTICAL
            code_done = BBACode.VERTICAL_DONE

        corrector = self.h_corrector if plane == 'H' else self.v_corrector
        initial_k0l = self.initial_h_k0l if plane == 'H' else self.initial_v_k0l
        data = self.H_data if plane == 'H' else self.V_data

        logger.debug('Setting corrector to under first value (k0 - 1.2 dk0) for hysteresis')
        interface.set(corrector, initial_k0l - 1.2 * data.dk0l)
        yield code

        k0_array = np.linspace(-data.dk0l, data.dk0l, self.n0) + initial_k0l

        get_orbit = self._interface.get_orbit
        for ii, k0_sp in enumerate(k0_array):

            # set next setpoint in corrector
            logger.debug(f'{ii+1}/{self.n0} Stepping to next corrector setpoint: {k0_sp*1e6:+.1f} murad')
            interface.set(corrector, k0_sp)
            yield code

            # correct vertical orbit?

            logger.debug('    Acquiring orbit data for quad zero')
            orbit_x, orbit_y, orbit_x_err, orbit_y_err = get_average_orbit(get_orbit=get_orbit, n_orbits=self.shots_per_orbit)
            data.raw_bpm_x_center.append(list(orbit_x))
            data.raw_bpm_y_center.append(list(orbit_y))
            data.raw_bpm_x_center_err.append(list(orbit_x_err))
            data.raw_bpm_y_center_err.append(list(orbit_y_err))

            # set "up" setpoint in quadrupole
            logger.debug(f'    Setting quadrupole to up setpoint: {self.initial_k1l + data.dk1l} 1/m')
            interface.set(self.quadrupole, self.initial_k1l + data.dk1l)
            yield code

            logger.debug('    Acquiring orbit data for quad up')
            orbit_x, orbit_y, orbit_x_err, orbit_y_err = get_average_orbit(get_orbit=get_orbit, n_orbits=self.shots_per_orbit)
            data.raw_bpm_x_up.append(list(orbit_x))
            data.raw_bpm_y_up.append(list(orbit_y))
            data.raw_bpm_x_up_err.append(list(orbit_x_err))
            data.raw_bpm_y_up_err.append(list(orbit_y_err))

            if self.bipolar:
                # set "down" setpoint in quadrupole
                logger.debug(f'    Setting quadrupole to down setpoint: {self.initial_k1l - data.dk1l}')
                interface.set(self.quadrupole, self.initial_k1l - data.dk1l)
                yield code

                logger.debug('    Acquiring orbit data for quad down')
                # orbit_x, orbit_y = get_pydoocs_orbit()
                orbit_x, orbit_y, orbit_x_err, orbit_y_err = get_average_orbit(get_orbit=get_orbit, n_orbits=self.shots_per_orbit)
                data.raw_bpm_x_down.append(list(orbit_x))
                data.raw_bpm_y_down.append(list(orbit_y))
                data.raw_bpm_x_down_err.append(list(orbit_x_err))
                data.raw_bpm_y_down_err.append(list(orbit_y_err))
            else:
                zero_orbit = list(orbit_x*0)
                data.raw_bpm_x_down.append(zero_orbit)
                data.raw_bpm_y_down.append(zero_orbit)
                data.raw_bpm_x_down_err.append(zero_orbit)
                data.raw_bpm_y_down_err.append(zero_orbit)


            # restore quadrupole to initial setpoint
            logger.debug(f'    Restoring quadrupole to initial setpoint: {self.initial_k1l}')
            interface.set(self.quadrupole, self.initial_k1l)

            logger.debug("")
        # restore corrector to initial setpoint
        interface.set(corrector, initial_k0l)

        #save data
        yield code_done

    def generate(self, interface: AbstractInterface):
        """
        step through the measurement.
        """
        self._interface = interface 

        timestamp = datetime.datetime.now().timestamp()
        # for restoring at the end
        self.initial_k1l = interface.get(self.quadrupole)
        if self.h_corrector is not None:
            self.initial_h_k0l = interface.get(self.h_corrector)
            self.H_data.initial_k0l = self.initial_h_k0l
            self.H_data.initial_k1 = self.initial_k1l
            self.H_data.timestamp = timestamp

        if self.v_corrector is not None:
            self.initial_v_k0l = interface.get(self.v_corrector)
            self.V_data.initial_k0l = self.initial_v_k0l
            self.V_data.initial_k1 = self.initial_k1l
            self.V_data.timestamp = timestamp

        self.print_init()

        if self.h_corrector is None:
            dk1 = self.V_data.dk1l
        elif self.v_corrector is None:
            dk1 = self.H_data.dk1l
        else:
            dk1 = max(self.H_data.dk1l, self.V_data.dk1l)

        for code in hysteresis_loop(self.quadrupole, interface, dk1, n_cycles=2, bipolar=self.bipolar):
            yield code

        if self.h_corrector is not None:
            for code in self.one_plane_loop('H'):
                yield code

        if self.v_corrector is not None:
            for code in self.one_plane_loop('V'):
                yield code

        yield BBACode.DONE

    def run(self, generator=None):
        if generator is None:
            generator = self.generate()
        for code in generator:
            logger.debug(f'    Got code: {code}')


class BBAAnalysis(BaseModel):
    offset: float
    offset_error: float

    slopes: list[float]
    centers: list[float]

    modulation: list[list[float]]
    position: list[list[float]]

    rejected_outliers: int
    rejected_slopes: int
    rejected_centers: int

    @classmethod
    def analyze(cls, data: BBAData):
        return BBAAnalysis()


def analyze_bba_data(data: BBAData):
    bpm_number = data.bpm_number
    nbpms = len(data.raw_bpm_x_center[0])
    orbits = np.full((data.n0, 2, nbpms), np.nan)
    bpm_pos = np.full((data.n0, 2), np.nan)
    for ii in range(data.n0):
        if data.plane == 'X':
            bpm_pos[ii, 0] = data.raw_bpm_x_up[ii][bpm_number]
            bpm_pos[ii, 1] = data.raw_bpm_x_down[ii][bpm_number]
            orbits[ii, 0] = np.array(data.raw_bpm_x_up[ii]) - np.array(data.raw_bpm_x_center[ii])
            orbits[ii, 1] = np.array(data.raw_bpm_x_down[ii]) - np.array(data.raw_bpm_x_center[ii])
        else:
            bpm_pos[ii, 0] = data.raw_bpm_y_up[ii][bpm_number]
            bpm_pos[ii, 1] = data.raw_bpm_y_down[ii][bpm_number]
            orbits[ii, 0] = np.array(data.raw_bpm_y_up[ii]) - np.array(data.raw_bpm_y_center[ii])
            orbits[ii, 1] = np.array(data.raw_bpm_y_down[ii]) - np.array(data.raw_bpm_y_center[ii])

    slopes, slopes_err, center, center_err = get_slopes_center(bpm_pos, orbits, data.dk1l)
    mask_bpm_outlier = reject_bpm_outlier(orbits)
    mask_slopes = reject_slopes(slopes)
    mask_center = reject_center_outlier(center)
    final_mask = np.logical_and(np.logical_and(mask_bpm_outlier, mask_slopes), mask_center)

    offset, offset_err = get_offset(center, center_err, final_mask)
    return offset, offset_err
