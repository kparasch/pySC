import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.simulated_commissioning import SimulatedCommissioning

logger = logging.getLogger(__name__)

def configure_injection(SC: "SimulatedCommissioning") -> None:
    injection_conf = dict.get(SC.configuration, 'injection', {})

    if 'from_design' in injection_conf:
        from_design = injection_conf['from_design']
    else:
        from_design = True

    if from_design:
        logger.info('Setting injection parameters from design twiss at the beginning of the lattice.')
        twiss = SC.lattice.twiss
        SC.injection.x = twiss['x'][0]
        SC.injection.px = twiss['px'][0]
        SC.injection.y = twiss['y'][0]
        SC.injection.py = twiss['py'][0]
        SC.injection.tau = twiss['tau'][0]
        SC.injection.delta = twiss['delta'][0]

        SC.injection.betx = twiss['betx'][0]
        SC.injection.alfx = twiss['alfx'][0]
        SC.injection.bety = twiss['bety'][0]
        SC.injection.alfy = twiss['alfy'][0]

    if 'emit_x' in injection_conf:
        SC.injection.gemit_x = float(injection_conf['emit_x'])
    else:
        logger.warning('emit_x not specified in injection configuration. Using default value of 1.')
    if 'emit_y' in injection_conf:
        SC.injection.gemit_y = float(injection_conf['emit_y'])
    else:
        logger.warning('emit_y not specified in injection configuration. Using default value of 1.')
    if 'bunch_length' in injection_conf:
        SC.injection.bunch_length = float(injection_conf['bunch_length'])
    else:
        logger.warning('bunch_length (in m, r.m.s.) not specified in injection configuration. Using default value of 1.')
    if 'energy_spread' in injection_conf:
        SC.injection.energy_spread = float(injection_conf['energy_spread'])
    else:
        logger.warning('energy_spread not specified in injection configuration. Using default value of 1.')

    for var in ['x', 'px', 'y', 'py', 'tau', 'delta', 'betx', 'alfx', 'bety', 'alfy']:
        if var in injection_conf:
            setattr(SC.injection, var, float(injection_conf[var]))

    for var in ['x_error_syst', 'px_error_syst', 'y_error_syst', 'py_error_syst', 'tau_error_syst', 'delta_error_syst',
                'x_error_stat', 'px_error_stat', 'y_error_stat', 'py_error_stat', 'tau_error_stat', 'delta_error_stat']:
        if var in injection_conf:
            setattr(SC.injection, var, float(injection_conf[var]))