from typing import Union, Optional, TYPE_CHECKING
import numpy as np
import logging

from .pySC_interface import pySCInjectionInterface, pySCOrbitInterface
from ..apps import measure_ORM, measure_dispersion

if TYPE_CHECKING:
    from ..core.simulated_commissioning import SimulatedCommissioning

logger = logging.getLogger(__name__)

def measure_TrajectoryResponseMatrix(SC: "SimulatedCommissioning", n_turns: int = 1,
                                     dkick: Union[float, list] = 1e-5, use_design: bool = False,
                                     normalize: bool = True, bipolar: bool = False) -> np.ndarray:
    logger.info(f'Measuring trajectory response matrix with {n_turns=}.')

    ### set inputs
    HCORR = SC.tuning.HCORR
    VCORR = SC.tuning.VCORR
    corrector_names = HCORR + VCORR

    interface = pySCInjectionInterface(SC=SC, n_turns=n_turns)
    interface.use_design = use_design

    generator = measure_ORM(interface=interface, corrector_names=corrector_names,
                            delta=dkick, bipolar=bipolar, skip_save=True)

    for code, measurement in generator:
        pass

    data = measurement.response_data
    if normalize:
        matrix = data.matrix 
    else:
        matrix = data.not_normalized_response_matrix

    return matrix

def measure_OrbitResponseMatrix(SC: "SimulatedCommissioning", HCORR: Optional[list] = None,
                                VCORR: Optional[list] = None, dkick: Union[float, list] = 1e-5,
                                use_design: bool = False, normalize: bool = True, bipolar: bool = True) -> np.ndarray:
    logger.info('Measuring orbit response matrix.')

    ### set inputs
    if HCORR is None:
        HCORR = SC.tuning.HCORR
    if VCORR is None:
        VCORR = SC.tuning.VCORR
    corrector_names = HCORR + VCORR

    interface = pySCOrbitInterface(SC=SC)
    interface.use_design = use_design

    generator = measure_ORM(interface=interface, corrector_names=corrector_names,
                            delta=dkick, bipolar=bipolar, skip_save=True)

    for code, measurement in generator:
        pass

    data = measurement.response_data
    if normalize:
        matrix = data.matrix
    else:
        matrix = data.not_normalized_response_matrix

    return matrix

def measure_RFFrequencyOrbitResponse(SC: "SimulatedCommissioning", delta_frf : float = 20, use_design: bool = False,
                                     normalize: bool = True, bipolar: bool = False) -> np.ndarray:
    logger.info('Measuring orbit response to RF frequency (dispersion).')
    interface = pySCOrbitInterface(SC=SC)
    interface.use_design = use_design

    generator = measure_dispersion(interface=interface, delta=delta_frf, bipolar=bipolar, skip_save=True)

    for code, measurement in generator:
        pass

    data = measurement.dispersion_data
    if normalize:
        response = np.concatenate(data.frequency_response)
    else:
        response = np.concatenate(data.not_normalized_frequency_response)

    return response
