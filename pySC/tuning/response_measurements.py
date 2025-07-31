from typing import Union, TYPE_CHECKING
import numpy as np
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeRemainingColumn

if TYPE_CHECKING:
    from ..core.new_simulated_commissioning import SimulatedCommissioning

progress = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeRemainingColumn(),
                   )

def response_loop(inputs, inputs_delta, get_output, settings):
    n_inputs = len(inputs)

    with progress:
        task_id = progress.add_task("Measuring RM", start=True, total=n_inputs)
        progress.update(task_id, total=n_inputs)

        reference = get_output()
        if np.any(np.isnan(reference)):
            raise ValueError('Initial output is NaN. Aborting. ')

        RM = np.full((len(reference), n_inputs), np.nan)

        for i, control in enumerate(inputs):
            ref_setpoint = settings.get(control)
            delta = inputs_delta[i]
            settings.set(control, ref_setpoint + delta)
            output = get_output()
            RM[:, i] = (output - reference)/delta
            settings.set(control, ref_setpoint)

            progress.update(task_id, completed=i+1, description=f'Measuring response of {control}...')
        progress.update(task_id, completed=n_inputs, description='Response measured.')
        progress.remove_task(task_id)

    return RM

def measure_TrajectoryResponseMatrix(SC: "SimulatedCommissioning", n_turns: int = 1, dkick: Union[float, list] = 1e-5, use_design: bool = False):
    print('Calculating response matrix')

    ### set inputs
    HCORR = SC.tuning.HCORR
    VCORR = SC.tuning.VCORR
    CORR = HCORR + VCORR

    n_CORR = len(CORR)

    if type(dkick) is float:
        kicks = np.ones(n_CORR, dtype=float) * dkick
    else:
        assert len(dkick) == n_CORR, f'ERROR: wrong length of dkick array provided. expected {n_CORR}, got {len(dkick)}'
        kicks = np.array(dkick)

    ### set function that gathers outputs
    def get_orbit():
        x,y = SC.bpm_system.capture_injection(n_turns=n_turns, bba=False, subtract_reference=False, use_design=use_design)
        return np.concat((x.flatten(order='F'), y.flatten(order='F')))

    ### specify to use "real" lattice or design
    magnet_settings = SC.design_magnet_settings if use_design else SC.magnet_settings

    ### measure the response matrix
    RM = response_loop(inputs=CORR, inputs_delta=kicks, get_output=get_orbit, settings=magnet_settings)

    return RM

def measure_OrbitResponseMatrix(SC: "SimulatedCommissioning", dkick: Union[float, list] = 1e-5, use_design: bool = False):
    print('Calculating response matrix')

    ### set inputs
    HCORR = SC.tuning.HCORR
    VCORR = SC.tuning.VCORR
    CORR = HCORR + VCORR

    n_CORR = len(CORR)

    if type(dkick) is float:
        kicks = np.ones(n_CORR, dtype=float) * dkick
    else:
        assert len(dkick) == n_CORR, f'ERROR: wrong length of dkick array provided. expected {n_CORR}, got {len(dkick)}'
        kicks = np.array(dkick)

    ### set function that gathers outputs
    def get_orbit():
        x,y = SC.bpm_system.capture_orbit(bba=False, subtract_reference=False, use_design=use_design)
        return np.concat((x.flatten(order='F'), y.flatten(order='F')))

    ### specify to use "real" lattice or design
    magnet_settings = SC.design_magnet_settings if use_design else SC.magnet_settings

    ### measure the response matrix
    RM = response_loop(inputs=CORR, inputs_delta=kicks, get_output=get_orbit, settings=magnet_settings)

    return RM

def measure_RFFrequencyOrbitResponse(SC: "SimulatedCommissioning", delta_frf : float = 20, rf_system_name: str = 'main', use_design: bool = False):

    rf_settings = SC.design_rf_settings if use_design else SC.rf_settings
    rf_system = rf_settings.systems[rf_system_name]

    ### function that gathers outputs
    def get_orbit():
        x,y = SC.bpm_system.capture_orbit(bba=False, subtract_reference=False, use_design=use_design)
        return np.concat((x.flatten(order='F'), y.flatten(order='F')))

    frf = rf_system.frequency
    
    xy0 = get_orbit()
    rf_system.set_frequency(frf + delta_frf)
    xy1 = get_orbit()
    rf_system.set_frequency(frf)
    response = (xy1 - xy0)/delta_frf

    return response


