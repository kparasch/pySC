from typing import Union
import numpy as np
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeRemainingColumn

progress = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeRemainingColumn(),
                   )

def measure_TrajectoryResponseMatrix(SC, n_turns: int = 1, dkick: Union[float, list] = 1e-5, use_design: bool = False):
    print('Calculating response matrix')

    HCORR = SC.tuning.HCORR
    VCORR = SC.tuning.VCORR
    CORR = HCORR + VCORR

    n_CORR = len(CORR)

    if type(dkick) is float:
        kicks = np.ones(n_CORR, dtype=float) * dkick
    else:
        assert len(dkick) == n_CORR, f'ERROR: wrong length of dkick array provided. expected {n_CORR}, got {len(dkick)}'
        kicks = np.array(dkick)

    def get_orbit():
        x,y = SC.bpm_system.capture_injection(n_turns=n_turns, bba=False, subtract_reference=False, use_design=use_design)
        return np.concat((x.flatten(order='F'), y.flatten(order='F')))

    magnet_settings = SC.design_magnet_settings if use_design else SC.magnet_settings

    RM = response_loop(inputs=CORR, inputs_delta=kicks, get_output=get_orbit, settings=magnet_settings)
#     with progress:
#         task_id = progress.add_task("Measuring RM", start=True, total=n_CORR)
#         progress.update(0, total=n_CORR)
# 
#         ref_trajectory_x, ref_trajectory_y = get_orbit()
#         if np.any(np.isnan(ref_trajectory_x)) or np.any(np.isnan(ref_trajectory_y)):
#             raise ValueError('Initial trajectory/orbit is NaN. Aborting. ')
# 
#         RM = np.full((n_CORR, len(ref_trajectory_x) + len(ref_trajectory_y)), np.nan)
# 
#         for i, control in enumerate(CORR):
#             ref_setpoint = magnet_settings.get(control)
#             kick = kicks[i]
#             magnet_settings.set(control, kick + ref_setpoint)
#             trajectory_x, trajectory_y = get_orbit()
#             RM[i, :n_BPM * n_turns] = trajectory_x - ref_trajectory_x
#             RM[i, n_BPM * n_turns:] = trajectory_y - ref_trajectory_y
#             magnet_settings.set(control, ref_setpoint)
# 
#             progress.update(task_id, completed=i+1, description=f'Measuring RM: {control}')

    return RM

def response_loop(inputs, inputs_delta, get_output, settings):
    n_inputs = len(inputs)

    with progress:
        task_id = progress.add_task("Measuring RM", start=True, total=n_inputs)
        progress.update(0, total=n_inputs)

        reference = get_output()
        if np.any(np.isnan(reference)):
            raise ValueError('Initial output is NaN. Aborting. ')

        RM = np.full((n_inputs, len(reference)), np.nan)

        for i, control in enumerate(inputs):
            ref_setpoint = settings.get(control)
            delta = inputs_delta[i]
            settings.set(control, ref_setpoint + delta)
            output = get_output()
            RM[i, :] = output - reference
            settings.set(control, ref_setpoint)

            progress.update(task_id, completed=i+1, description=f'Measuring response of {control}...')
        progress.update(task_id, completed=n_inputs, description='Response measured.')

    return RM
