import numpy as np
import logging
from pydantic import BaseModel
from typing import Optional, Callable
from enum import IntEnum

logger = logging.getLogger(__name__)

class MeasurementCode(IntEnum):
    INITIALIZED = 0
    HYSTERESIS = 1
    HYSTERESIS_DONE = 2

def hysteresis_loop(name, settings, delta, n_cycles=1, bipolar=True):
    sp0 = settings.get(name)

    logger.debug(f'Hysteresis loop started for {name}.')
    logger.debug('    Going up to (sp0 + delta)')

    for _ in range(n_cycles):
        settings.set(sp0 + delta)
        yield MeasurementCode.HYSTERESIS
        if bipolar:
            logger.debug('    Going down to (sp0 - delta)')
            settings.set(sp0 - delta)
        else:
            logger.debug('    Going down to (sp0)')
            settings.set(sp0)
        yield MeasurementCode.HYSTERESIS

    if bipolar:
        logger.debug('    Going back to (sp0 - delta)')
        settings.set(sp0)
    yield MeasurementCode.HYSTERESIS_DONE


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
