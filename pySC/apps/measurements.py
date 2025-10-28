import logging
import numpy as np

from ..tuning.response_matrix import ResponseMatrix
from typing import Optional, Callable, Union
from .bba import BBA_Measurement, BBACode

logger = logging.getLogger(__name__)

def orbit_correction(interface, response_matrix: ResponseMatrix, method='svd_cutoff',
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

def measure_bba(interface, bpm_name, config: dict, shots_per_orbit: int = 1,
                n_corr_steps: int = 7, bipolar: bool = True, skew_quad: bool = False,
                skip_save: bool = False):

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
                                  quad_is_skew=skew_quad
                                 )

    generator = measurement.generate(interface=interface)

    # run measurement loop
    for code in generator:
        logger.debug(f'    Got code: {code}')
        if not skip_save and code is BBACode.HORIZONTAL_DONE:
            measurement.H_data.save()
        if not skip_save and code is BBACode.VERTICAL_DONE:
            measurement.V_data.save()
        # yield code ## we should probably yield

    H_data = measurement.H_data
    V_data = measurement.V_data
    ## analyze data here ? 
    return 