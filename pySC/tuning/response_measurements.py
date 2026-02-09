from typing import Union, Optional, TYPE_CHECKING
import numpy as np
from .pySC_interface import pySCInjectionInterface, pySCOrbitInterface
from ..apps import measure_ORM

if TYPE_CHECKING:
    from ..core.new_simulated_commissioning import SimulatedCommissioning

def measure_TrajectoryResponseMatrix(SC: "SimulatedCommissioning", n_turns: int = 1, dkick: Union[float, list] = 1e-5, use_design: bool = False, normalize: bool = True, bipolar: bool = False):
    print('Calculating response matrix')

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

def measure_OrbitResponseMatrix(SC: "SimulatedCommissioning", HCORR: Optional[list] = None, VCORR: Optional[list] = None, dkick: Union[float, list] = 1e-5, use_design: bool = False, normalize: bool = True, bipolar: bool = True):
    print('Calculating response matrix')

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

def measure_RFFrequencyOrbitResponse(SC: "SimulatedCommissioning", delta_frf : float = 20, rf_system_name: str = 'main', use_design: bool = False, normalize: bool = True, bipolar: bool = False):

    rf_settings = SC.design_rf_settings if use_design else SC.rf_settings
    rf_system = rf_settings.systems[rf_system_name]

    ### function that gathers outputs
    def get_orbit():
        x,y = SC.bpm_system.capture_orbit(bba=False, subtract_reference=False, use_design=use_design)
        return np.concat((x.flatten(order='F'), y.flatten(order='F')))

    frf = rf_system.frequency
    if bipolar:
        step = delta_frf / 2
        rf_system.set_frequency(frf - step)
        xy0 = get_orbit()
    else:
        step = delta_frf
        xy0 = get_orbit()
    rf_system.set_frequency(frf + step)
    xy1 = get_orbit()
    rf_system.set_frequency(frf)
    response = (xy1 - xy0)
    if normalize:
        response /= delta_frf

    return response
