import numpy as np
import logging
from .response_matrix import ResponseMatrix

logger = logging.getLogger(__name__)

def orbit_correction(get_orbit, settings, response_matrix: ResponseMatrix, correctors, method='svd_cutoff', parameter=0, reference=None, gain=1, apply=False):

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
