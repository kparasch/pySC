import logging
import numpy as np
from typing import Optional, Generator, Union
from pathlib import Path

from ..tuning.response_matrix import ResponseMatrix
from .bba import BBA_Measurement, BBACode
from .response import ResponseMeasurement, ResponseCode
from .dispersion import DispersionMeasurement, DispersionCode
from .interface import AbstractInterface

logger = logging.getLogger(__name__)

def orbit_correction(interface: AbstractInterface, response_matrix: ResponseMatrix, method='svd_cutoff',
                     parameter: Union[int,float] = 0, reference: Optional[np.ndarray] = None,
                     gain: float = 1, apply: bool = False):

    correctors = response_matrix.input_names
    assert correctors is not None, 'Corrector names are undefined in the response matrix'

    if not apply and gain != 1:
        logger.warning("Gain is set but apply is False, gain will have no effect.")

    orbit_x, orbit_y = interface.get_orbit()
    orbit = np.concat((orbit_x.flatten(order='F'), orbit_y.flatten(order='F')))

    if reference is not None:
        assert len(reference) == len(orbit), "Reference orbit has wrong length"
        orbit -= reference

    trim_list = -response_matrix.solve(orbit, method=method, parameter=parameter)

    trims = {corr: trim for corr, trim in zip(correctors, trim_list) if trim != 0}

    if apply:
        data = interface.get_many(correctors)
        for i, corr in enumerate(correctors):
            data[corr] += trim_list[i] * gain
        interface.set_many(data)

    return trims

def measure_bba(interface: AbstractInterface, bpm_name, config: dict, shots_per_orbit: int = 1,
                n_corr_steps: int = 7, bipolar: bool = True, skip_save: bool = False) -> Generator:

    folder_to_save = None
    if folder_to_save is None:
        folder_to_save = Path('data')

    if not skip_save:
        assert folder_to_save.exists(), f'Path {folder_to_save.resolve()} does not exist.'
        assert folder_to_save.is_dir(), f'Path {folder_to_save.resolve()} is not a directory.'

    keys = config.keys()
    for word in ['QUAD', 'HCORR', 'HCORR_delta', 'QUAD_dk_H', 'VCORR', 'VCORR_delta', 'QUAD_dk_V', 'QUAD_is_skew', 'number']:
        assert word in keys, f'{word} is not in configuration.'

    measurement = BBA_Measurement(bpm=bpm_name,
                                  quadrupole=config['QUAD'],
                                  h_corrector=config['HCORR'],
                                  v_corrector=config['VCORR'],
                                  dk0l_x=config['HCORR_delta'],
                                  dk1l_x=config['QUAD_dk_H'],
                                  dk0l_y=config['VCORR_delta'],
                                  dk1l_y=config['QUAD_dk_V'],
                                  quad_is_skew=config['QUAD_is_skew'],
                                  n0=n_corr_steps,
                                  bpm_number=config['number'],
                                  shots_per_orbit=shots_per_orbit,
                                  bipolar=bipolar,
                                 )

    generator = measurement.generate(interface=interface)

    # run measurement loop
    for code in generator:
        logger.debug(f'    Got code: {code}')
        if not skip_save and code is BBACode.HORIZONTAL_DONE:
            measurement.H_data.save(folder_to_save=folder_to_save)
        if not skip_save and code is BBACode.VERTICAL_DONE:
            measurement.V_data.save(folder_to_save=folder_to_save)
        yield code, measurement 

def measure_ORM(interface: AbstractInterface, corrector_names: list[str], delta: Union[float, list[float]],
                shots_per_orbit: int = 1, bipolar=True, skip_save: bool = False, save_every: Optional[int] = None):

    folder_to_save = None
    if folder_to_save is None:
        folder_to_save = Path('data')

    if not skip_save:
        assert folder_to_save.exists(), f'Path {folder_to_save.resolve()} does not exist.'
        assert folder_to_save.is_dir(), f'Path {folder_to_save.resolve()} is not a directory.'

    orbit_x, orbit_y = interface.get_orbit()
    n_outputs = len(orbit_x) + len(orbit_y)

    measurement = ResponseMeasurement(inputs_delta=delta,
                                      shots_per_orbit=shots_per_orbit,
                                      bipolar=bipolar,
                                      input_names=corrector_names,
                                      n_outputs=n_outputs)

    generator = measurement.generate(interface=interface, get_output=interface.get_orbit)
    for code in generator:
        if not skip_save and code is ResponseCode.MEASURING:
            if save_every is not None and measurement.last_number % save_every == 0:
                measurement.response_data.save(folder_to_save=folder_to_save)

        if not skip_save and code is ResponseCode.DONE:
            measurement.response_data.save(folder_to_save=folder_to_save)
        yield code, measurement


def measure_dispersion(interface: AbstractInterface, delta: float, shots_per_orbit: int = 1,
                       bipolar=True, skip_save: bool = False) -> Generator:

    folder_to_save = None
    if folder_to_save is None:
        folder_to_save = Path('data')

    if not skip_save:
        assert folder_to_save.exists(), f'Path {folder_to_save.resolve()} does not exist.'
        assert folder_to_save.is_dir(), f'Path {folder_to_save.resolve()} is not a directory.'

    measurement = DispersionMeasurement(delta=delta,
                                        shots_per_orbit=shots_per_orbit,
                                        bipolar=bipolar)

    generator = measurement.generate(interface=interface, get_output=interface.get_orbit)
    for code in generator:
        if not skip_save and code is DispersionCode.DONE:
            measurement.dispersion_data.save(folder_to_save=folder_to_save)
        yield code, measurement
