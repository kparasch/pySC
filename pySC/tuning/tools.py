import numpy as np
import logging
from .response_matrix import ResponseMatrix
from pydantic import BaseModel
from typing import Optional, Callable
import datetime
from ..utils.file_tools import dict_to_h5
from .bba import BBA_Measurement ## circular import :(

logger = logging.getLogger(__name__)

def orbit_correction(get_orbit: Callable, settings, response_matrix: ResponseMatrix, correctors, method='svd_cutoff', parameter=0, reference=None, gain=1, apply=False):

    if not apply and gain != 1:
        logger.warning("Gain is set but apply is False, gain will have no effect.")

    orbit_x, orbit_y = get_orbit()
    orbit = np.concat((orbit_x.flatten(order='F'), orbit_y.flatten(order='F')))

    if reference is not None:
        assert len(reference) == len(orbit), "Reference orbit has wrong length"
        orbit -= reference

    trims = response_matrix.solve(orbit, method=method, parameter=parameter)

    if apply:
        data = settings.get_many(correctors)
        for i, corr in enumerate(correctors):
            data[corr] += trims[i] * gain
        settings.set_many(data)

    return trims

def hysteresis_loop(name, settings, delta, n_cycles=1, bipolar=True):
    sp0 = settings.get(name)

    logger.debug(f'Hysteresis loop started for {name}.')
    logger.debug('    Going up to (sp0 + delta)')

    for _ in range(n_cycles):
        settings.set(sp0 + delta)
        yield 0
        if bipolar:
            logger.debug('    Going down to (sp0 - delta)')
            settings.set(sp0 - delta)
        else:
            logger.debug('    Going down to (sp0)')
            settings.set(sp0)
        yield 0

    if bipolar:
        logger.debug('    Going back to (sp0 - delta)')
        settings.set(sp0)
    yield 1


def get_average_orbit(get_orbit: Callable, n_orbits: int = 10):
    orbit_x, orbit_y = get_orbit()
    all_orbit_x = np.zeros((len(orbit_x), n_orbits))
    all_orbit_y = np.zeros((len(orbit_y), n_orbits))

    all_orbit_x[:,0] = orbit_x
    all_orbit_y[:,0] = orbit_y
    for ii in range(1, n_orbits):
        all_orbit_x[:, ii], all_orbit_y[:, ii] = get_orbit()

    mean_orbit_x = np.mean(all_orbit_x, axis=1)
    mean_orbit_y = np.mean(all_orbit_y, axis=1)
    std_orbit_x = np.std(all_orbit_x, axis=1)
    std_orbit_y = np.std(all_orbit_y, axis=1)
    return mean_orbit_x, mean_orbit_y, std_orbit_x, std_orbit_y


def orbit_bba(get_orbit, settings, bpm_name, plane, config, shots_per_orbit: int = 1,
              n_corr_steps: int = 7, bipolar: bool = True, skip_save: bool = False):

    measurement = BBA_Measurement(bpm=bpm_name,
                                  quadrupole=config['QUAD'],
                                  h_corrector=config['HCORR'],
                                  v_corrector=config['VCORR'],
                                  dk0l_x=config['HCORR_delta'],
                                  dk1_x=config['QUAD_dk_H'],
                                  dk0l_y=config['VCORR_delta'],
                                  dk1_y=config['QUAD_dk_V'],
                                  n0=n_corr_steps,
                                  bpm_number=config['number'],
                                  shots_per_orbit=shots_per_orbit,
                                  bipolar=bipolar,
                                  skip_save=skip_save
                                 )

    measurement.generate(get_orbit=get_orbit, settings=settings)
    measurement.run() # ??

    H_data = measurement.H_data
    V_data = measurement.V_data
    ## analyze data here ? 
    return 

