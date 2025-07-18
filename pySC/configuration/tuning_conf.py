from ..core.new_simulated_commissioning import SimulatedCommissioning
import numpy as np

def sort_correctors(SC: SimulatedCommissioning, control_names: list[str]) -> list[str]:
    names = [control_name.split('/')[0] for control_name in control_names]
    indices = [SC.magnet_settings.magnets[name].sim_index for name in names]
    argsort = np.argsort(indices).tolist()
    sorted_control_names = [control_names[i] for i in argsort]
    return sorted_control_names

def configure_correctors(SC: SimulatedCommissioning, config_dict: dict) -> list[str]:
    CORR = []
    for category in config_dict:
        array_name, component = category.copy().popitem()
        for control_name in SC.control_arrays[array_name]:
            if component == control_name.split('/')[1]:
                CORR.append(control_name)
    return CORR

def configure_tuning(SC: SimulatedCommissioning) -> None:
    tuning_conf = dict.get(SC.configuration, 'tuning', {})

    HCORR = configure_correctors(SC, config_dict=tuning_conf['HCORR'])
    VCORR = configure_correctors(SC, config_dict=tuning_conf['VCORR'])

    if 'sort_correctors' in tuning_conf and tuning_conf['sort_correctors']:
        HCORR = sort_correctors(SC, HCORR)
        VCORR = sort_correctors(SC, VCORR)

    SC.tuning.HCORR = HCORR
    SC.tuning.VCORR = VCORR

    if 'model_RM_folder' in tuning_conf:
        SC.tuning.RM_folder = tuning_conf['model_RM_folder']