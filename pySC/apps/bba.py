from pydantic import BaseModel, PrivateAttr, ConfigDict
from typing import Optional, ClassVar
import datetime
import logging
import numpy as np
from pathlib import Path

from .codes import BBACode
from ..utils.file_tools import dict_to_h5
from .tools import get_average_orbit
from .interface import AbstractInterface
from ..core.types import NPARRAY

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
    plane: str = None

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
            self.H_data = BBAData(plane='H', bpm=self.bpm, quadrupole=self.quadrupole, bpm_number=self.bpm_number,
                                  corrector=self.h_corrector, dk0l=self.dk0l_x, dk1l=self.dk1l_x,
                                  n0=self.n0, shots_per_orbit=self.shots_per_orbit, bipolar=self.bipolar,
                                  skew_quad=self.quad_is_skew)
        if self.v_corrector is not None:
            self.V_data = BBAData(plane='V', bpm=self.bpm, quadrupole=self.quadrupole, bpm_number=self.bpm_number,
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

    def generate(self, interface: AbstractInterface, plane: Optional[str] = None, skip_cycle: bool = False):
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

        if not skip_cycle:
            for code in hysteresis_loop(self.quadrupole, interface, dk1, n_cycles=2, bipolar=self.bipolar):
                yield code

        if (plane is None or plane == 'H') and self.h_corrector is not None:
            for code in self.one_plane_loop('H'):
                yield code

        if (plane is None or plane == 'V') and self.v_corrector is not None:
            for code in self.one_plane_loop('V'):
                yield code

        yield BBACode.DONE

    # def run(self, generator=None):
    #     if generator is None:
    #         generator = self.generate()
    #     for code in generator:
    #         logger.debug(f'    Got code: {code}')

def prep_ios(data: BBAData, n_downstream: Optional[int] = None):
    bpm_number = data.bpm_number
    bpm_position = np.full((data.n0), np.nan)

    if n_downstream is not None:
        induced_orbit_shift = np.full((data.n0, n_downstream), np.nan)
        start = bpm_number
        end = bpm_number + n_downstream
    else:
        n_bpm = len(data.raw_bpm_x_center[0])
        induced_orbit_shift = np.full((data.n0, n_bpm), np.nan)
        start = 0
        end = n_bpm

    x_up = np.array(data.raw_bpm_x_up)
    y_up = np.array(data.raw_bpm_y_up)
    x_center = np.array(data.raw_bpm_x_center)
    y_center = np.array(data.raw_bpm_y_center)
    if data.bipolar:
        k1_arr = [-data.dk1l, 0, data.dk1l]
        x_down = np.array(data.raw_bpm_x_down)
        y_down = np.array(data.raw_bpm_y_down)
        all_x = np.array([x_down[:, start:end], x_center[:,start:end], x_up[:, start:end]])
        all_y = np.array([y_down[:, start:end], y_center[:,start:end], y_up[:, start:end]])
    else:
        k1_arr = [0, data.dk1l]
        all_x = np.array([x_center[:,start:end], x_up[:, start:end]])
        all_y = np.array([y_center[:,start:end], y_up[:, start:end]])

    for ii in range(data.n0):
        if data.plane == 'H':
            bpm_position[ii] = np.mean(all_x[:, ii, bpm_number - start])
            if data.skew_quad:
                induced_orbit_shift[ii] = np.polyfit(k1_arr, all_y[:,ii], 1)[0] * data.dk1l
            else:
                induced_orbit_shift[ii] = np.polyfit(k1_arr, all_x[:,ii], 1)[0] * data.dk1l
        else:
            bpm_position[ii] = np.mean(all_y[:, ii, bpm_number - start])
            if data.skew_quad:
                induced_orbit_shift[ii] = np.polyfit(k1_arr, all_x[:,ii], 1)[0] * data.dk1l
            else:
                induced_orbit_shift[ii] = np.polyfit(k1_arr, all_y[:,ii], 1)[0] * data.dk1l
    return bpm_position, induced_orbit_shift

def reject_bpm_outlier(induced_orbit_shift: np.ndarray, bpm_outlier_sigma: float) -> np.ndarray[bool]:
    n_bpms = induced_orbit_shift.shape[1]
    mask = np.ones(n_bpms, dtype=bool)
    for bpm in range(n_bpms):
        data = induced_orbit_shift[:, bpm]
        if np.any(data - np.mean(data) > bpm_outlier_sigma * np.std(data)):
            mask[bpm] = False
    return mask

def reject_slopes(slopes: np.ndarray, slope_cutoff: float) -> np.ndarray[bool]:
    max_slope = np.nanmax(np.abs(slopes))
    mask = np.abs(slopes) > slope_cutoff * max_slope
    return mask

def reject_center_outlier(center: np.ndarray, center_cutoff: float) -> np.ndarray[bool]:
    mean = np.nanmean(center)
    std = np.nanstd(center)
    mask =  abs(center - mean) < center_cutoff * std
    return mask

class BBAAnalysis(BaseModel):
    offset: float
    offset_error: float

    slopes: NPARRAY
    centers: NPARRAY

    slopes_err: NPARRAY
    centers_err: NPARRAY

    induced_orbit_shift: NPARRAY
    bpm_position: NPARRAY

    mask_accepted: NPARRAY

    n_downstream: Optional[int]

    rejected_outliers: int
    rejected_slopes: int
    rejected_centers: int

    bpm_outlier_sigma: float
    slope_cutoff: float
    center_cutoff: float

    default_bpm_outlier_sigma: ClassVar[float] = 6 # number of sigma
    default_slope_cutoff: ClassVar[float] = 0.10 # of max slope
    default_center_cutoff: ClassVar[int] = 1 # number of sigma

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def analyze(cls, data: BBAData, n_downstream: Optional[int] = None, bpm_outlier_sigma: Optional[float] = None,
                slope_cutoff: Optional[float] = None, center_cutoff: Optional[float] = None):

        if bpm_outlier_sigma is None:
            bpm_outlier_sigma = cls.default_bpm_outlier_sigma

        if slope_cutoff is None:
            slope_cutoff = cls.default_slope_cutoff

        if center_cutoff is None:
            center_cutoff = cls.default_center_cutoff

        bpm_position, induced_orbit_shift = prep_ios(data=data, n_downstream=n_downstream)

        p, pcov = np.polyfit(bpm_position, induced_orbit_shift, 1, cov=True)
        slopes = p[0]
        centers = - p[1] / p[0]
        slopes_err = np.sqrt(pcov[0,0])
        centers_err = np.sqrt(centers ** 2 * (pcov[0,0] / p[0]**2 + pcov[1,1] / p[1] ** 2 - 2 * pcov[0, 1] / p[0] / p[1]))

        mask_bpm_outlier = reject_bpm_outlier(induced_orbit_shift, bpm_outlier_sigma)
        mask_slopes = reject_slopes(slopes, slope_cutoff)
        mask_center = reject_center_outlier(centers, center_cutoff)
        mask_accepted = np.logical_and(np.logical_and(mask_bpm_outlier, mask_slopes), mask_center)

        # calculate offset as a weighted average of the centers, with weights equal to the absolute slope
        cc = centers[mask_accepted]
        ww = np.abs(slopes[mask_accepted])
        cc_err = centers_err[mask_accepted]
        ww_err = slopes_err[mask_accepted]

        CS = np.sum(ww * cc)
        S = np.sum(ww) 
        VS = np.sum(ww_err**2) # variance of S
        VCS = np.sum(cc**2 * ww_err**2 + ww**2 * cc_err**2) # variance of CS
        VO = ( VCS / CS**2 + VS / S**2) # (variance of offset) / offset**2

        offset = CS / S # offset = CS / S, average of centers with abs(slopes) as weights
        offset_error = offset * np.sqrt(VO)

        result = BBAAnalysis(
                             offset=offset,
                             offset_error=offset_error,
                             slopes=slopes,
                             centers=centers,
                             slopes_err=slopes_err,
                             centers_err=centers_err,
                             induced_orbit_shift=induced_orbit_shift,
                             bpm_position=bpm_position,
                             mask_accepted=mask_accepted,
                             n_downstream=n_downstream,
                             rejected_outliers=sum(~mask_bpm_outlier),
                             rejected_slopes=sum(~mask_slopes),
                             rejected_centers=sum(~mask_center),
                             total_rejections=sum(~mask_accepted),
                             bpm_outlier_sigma=bpm_outlier_sigma,
                             slope_cutoff=slope_cutoff,
                             center_cutoff=center_cutoff,
                            )
        return result
