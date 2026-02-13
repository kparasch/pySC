import logging

from ..core.simulated_commissioning import SimulatedCommissioning
from ..core.rfsettings import RFCavity, RFSystem
from .general import get_error, get_indices_and_names

logger = logging.getLogger(__name__)

def configure_rf(SC: SimulatedCommissioning) -> None:
    rf_conf = dict.get(SC.configuration, 'rf', {})
    error_table = dict.get(SC.configuration, 'error_table', {}) # defaults to empty error_table if not declared

    if 'main' not in rf_conf:
        logger.warning('"main" rf system was not found in the configuration file.')

    for rf_category in rf_conf.keys():
        rf_category_conf = rf_conf[rf_category]

        indices, cavity_names = get_indices_and_names(SC, rf_category, rf_category_conf)

        total_voltage = 0
        system_phase = None
        system_frequency = None
        for index, name in zip(indices, cavity_names):
            voltage, phase, frequency = SC.lattice.get_cavity_voltage_phase_frequency(index)

            if system_frequency is None or system_phase is None:
                system_phase = phase
                system_frequency = frequency

            total_voltage += voltage
            phase_delta = phase - system_phase
            frequency_delta = frequency - system_frequency

            if phase_delta != 0:
                logger.warning(f'Cavities in the {rf_category} rf system do not have the same phase.')
            if frequency_delta != 0:
                logger.warning(f'Cavities in the {rf_category} rf system do not have the same frequency.')

            cavity = RFCavity(sim_index=index, phase_delta=phase_delta, frequency_delta=frequency_delta)
            design_cavity = RFCavity(sim_index=index, phase_delta=phase_delta, frequency_delta=frequency_delta, to_design=True)

            if 'voltage' in rf_conf:
                sig = get_error(rf_conf['voltage'], error_table=error_table)
                cavity.voltage_error = SC.rng.normal_trunc(0, sig)

            if 'phase' in rf_conf:
                sig = get_error(rf_conf['phase'], error_table=error_table)
                cavity.phase_error = SC.rng.normal_trunc(0, sig)

            if 'frequency' in rf_conf:
                sig = get_error(rf_conf['frequncy'], error_table=error_table)
                cavity.frequency_error = SC.rng.normal_trunc(0, sig)

            SC.rf_settings.cavities[name] = cavity
            SC.design_rf_settings.cavities[name] = design_cavity

        SC.rf_settings.systems[rf_category] = RFSystem(cavities=cavity_names, voltage=total_voltage,
                                                       phase=system_phase, frequency=system_frequency)
        SC.design_rf_settings.systems[rf_category] = RFSystem(cavities=cavity_names, voltage=total_voltage,
                                                       phase=system_phase, frequency=system_frequency)

        SC.rf_settings.systems[rf_category]._parent = SC.rf_settings
        SC.design_rf_settings.systems[rf_category]._parent = SC.design_rf_settings
        for name in cavity_names:
            SC.rf_settings.cavities[name]._parent_system = SC.rf_settings.systems[rf_category]
            SC.design_rf_settings.cavities[name]._parent_system = SC.design_rf_settings.systems[rf_category]

        SC.rf_settings.systems[rf_category].trigger_update()
        SC.design_rf_settings.systems[rf_category].trigger_update()
