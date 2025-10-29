from typing import Union, Optional, TYPE_CHECKING
import numpy as np
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeRemainingColumn

if TYPE_CHECKING:
    from ..core.new_simulated_commissioning import SimulatedCommissioning


DISABLE_RICH = False

rich_progress = Progress(
                         TextColumn("[progress.description]{task.description}"),
                         BarColumn(),
                         MofNCompleteColumn(),
                         TimeRemainingColumn(),
                        )

def no_rich_progress():
    from contextlib import nullcontext
    progress = nullcontext()
    progress.add_task = lambda *args, **kwargs: 1
    progress.update = lambda *args, **kwargs: None
    progress.remove_task = lambda *args, **kwargs: None
    return progress

def response_loop(inputs, inputs_delta, get_output, settings, normalize=True, bipolar=False):
    n_inputs = len(inputs)

    if DISABLE_RICH:
        progress = no_rich_progress()
    else:
        progress = rich_progress

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
            if bipolar:
                step = delta/2
                settings.set(control, ref_setpoint - step)
                reference = get_output()
            else:
                step = delta
            settings.set(control, ref_setpoint + step)
            output = get_output()

            RM[:, i] = (output - reference)
            if normalize:
                RM[:, i] /= delta
            settings.set(control, ref_setpoint)

            progress.update(task_id, completed=i+1, description=f'Measuring response of {control}...')
        progress.update(task_id, completed=n_inputs, description='Response measured.')
        progress.remove_task(task_id)

    return RM

def measure_TrajectoryResponseMatrix(SC: "SimulatedCommissioning", n_turns: int = 1, dkick: Union[float, list] = 1e-5, use_design: bool = False, normalize: bool = True, bipolar: bool = False):
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
    RM = response_loop(inputs=CORR, inputs_delta=kicks, get_output=get_orbit, settings=magnet_settings, normalize=normalize, bipolar=bipolar)

    return RM

def measure_OrbitResponseMatrix(SC: "SimulatedCommissioning", HCORR: Optional[list] = None, VCORR: Optional[list] = None, dkick: Union[float, list] = 1e-5, use_design: bool = False, normalize: bool = True, bipolar: bool = True):
    print('Calculating response matrix')

    ### set inputs
    if HCORR is None:
        HCORR = SC.tuning.HCORR
    if VCORR is None:
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
    RM = response_loop(inputs=CORR, inputs_delta=kicks, get_output=get_orbit, settings=magnet_settings, normalize=normalize, bipolar=bipolar)

    return RM

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


