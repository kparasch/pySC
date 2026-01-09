from ..core.new_simulated_commissioning import SimulatedCommissioning
from ..core.control import IndivControl
import numpy as np

def sort_controls(SC: SimulatedCommissioning, control_names: list[str]) -> list[str]:
    magnet_names = []
    for control_name in control_names:
        control = SC.magnet_settings.controls[control_name]
        if type(control.info) is IndivControl:
            magnet_name = control.info.magnet_name
        else:
            raise NotImplementedError(f"{control} is of type {type(control.info).__name__} which is not implemented.")
        magnet_names.append(magnet_name)
    indices = [SC.magnet_settings.magnets[name].sim_index for name in magnet_names]
    argsort = np.argsort(indices).tolist()
    sorted_control_names = [control_names[i] for i in argsort]
    return sorted_control_names

def configure_family(SC: SimulatedCommissioning, config_dict: dict) -> list[str]:
    family = []
    for category in config_dict:
        array_name, component = category.copy().popitem()
        for control_name in SC.control_arrays[array_name]:
            if component == control_name.split('/')[1]:
                family.append(control_name)
    return family

def configure_tuning(SC: SimulatedCommissioning) -> None:
    tuning_conf = dict.get(SC.configuration, 'tuning', {})

    if 'HCORR' in tuning_conf:
        HCORR = configure_family(SC, config_dict=tuning_conf['HCORR'])
        HCORR = sort_controls(SC, HCORR)
        SC.tuning.HCORR = HCORR

    # if 'sort_correctors' in tuning_conf and tuning_conf['sort_correctors']:
    if 'VCORR' in tuning_conf:
        VCORR = configure_family(SC, config_dict=tuning_conf['VCORR'])
        VCORR = sort_controls(SC, VCORR)
        SC.tuning.VCORR = VCORR

    if 'model_RM_folder' in tuning_conf:
        SC.tuning.RM_folder = tuning_conf['model_RM_folder']

    if 'multipoles' in tuning_conf:
        multipoles = configure_family(SC, config_dict=tuning_conf['multipoles'])
        multipoles = sort_controls(SC, multipoles)
        SC.tuning.multipoles = multipoles

    if 'bba_magnets' in tuning_conf:
        bba_magnets = configure_family(SC, config_dict=tuning_conf['bba_magnets'])
        bba_magnets = sort_controls(SC, bba_magnets)
        SC.tuning.bba_magnets = bba_magnets

    if 'c_minus' in tuning_conf:
        c_minus_conf = tuning_conf['c_minus']
        if 'controls' in c_minus_conf:
            c_minus_controls = configure_family(SC, config_dict=c_minus_conf['controls'])
            c_minus_controls = sort_controls(SC, c_minus_controls)
            SC.tuning.c_minus.controls = c_minus_controls