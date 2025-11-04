import logging
import numpy as np
from typing import Optional, Generator, Union
from pathlib import Path

from ..tuning.response_matrix import ResponseMatrix
from .bba import BBA_Measurement, BBACode
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
                                  dk1_x=config['QUAD_dk_H'],
                                  dk0l_y=config['VCORR_delta'],
                                  dk1_y=config['QUAD_dk_V'],
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

    H_data = measurement.H_data
    V_data = measurement.V_data
    ## analyze data here ? 
    yield BBACode.DONE, measurement