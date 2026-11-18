from ..core.simulated_commissioning import SimulatedCommissioning
from ..core.control import IndivControl, SeriesControl
from .general import get_indices_and_names
import numpy as np
import logging

logger = logging.getLogger(__name__)

def sort_controls(SC: SimulatedCommissioning, control_names: list[str]) -> list[str]:
    magnet_names = []
    for control_name in control_names:
        control = SC.magnet_settings.controls[control_name]
        if type(control.info) is IndivControl:
            magnet_name = control.info.magnet_name
        elif type(control.info) is SeriesControl:
            magnet_name = control.info.magnet_names[0]
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
            control = SC.design_magnet_settings.controls[control_name]
            control_component = f"{control.info.component}{control.info.order}"
            if control.info.is_integrated:
                control_component = f"{control_component}L"
            if component == control_component:
                family.append(control_name)
    return family

def resolve_controls(SC: SimulatedCommissioning, config_name: str, config: dict | list) -> list[str]:
    """Resolve category/component entries or a regex/mapping plus component.

    Selecting any member of a series selects its shared control. Knobs are
    excluded. Results are unique and sorted by the existing lattice ordering.
    """
    if isinstance(config, list):
        for entry in config:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError("Each controls entry must contain one category: component pair.")
        controls = configure_family(SC, config)
    elif isinstance(config, dict):
        if 'component' not in config:
            raise ValueError("component is missing from controls selection.")
        if ('regex' in config) == ('mapping' in config):
            raise ValueError("Controls selection requires exactly one of regex or mapping.")
        component = config['component']
        indices, _ = get_indices_and_names(SC, "", config, verbose=False)
        settings = SC.magnet_settings
        controls = []
        for index in indices:
            magnet_name = settings.index_mapping.get(index)
            if magnet_name is None:
                continue
            for control in settings.magnets[magnet_name].controls:
                info = control.info
                if not isinstance(info, (IndivControl, SeriesControl)):
                    continue
                control_component = f"{info.component}{info.order}"
                if info.is_integrated:
                    control_component += 'L'
                if control_component == component:
                    controls.append(control.name)
    else:
        raise TypeError("Controls selection must be a dictionary or a list.")

    controls = list(dict.fromkeys(controls)) # remove duplicate controls
    controls = sort_controls(SC, controls)
    logger.info(f"Found {len(controls)} controls for tuning/{config_name}.")
    return controls

def configure_tuning(SC: SimulatedCommissioning) -> None:
    tuning_conf = dict.get(SC.configuration, 'tuning', {})

    if 'HCORR' in tuning_conf:
        SC.tuning.HCORR = resolve_controls(SC, "HCORR", tuning_conf["HCORR"])

    if 'VCORR' in tuning_conf:
        SC.tuning.VCORR = resolve_controls(SC, "VCORR", tuning_conf["VCORR"])

    if 'model_RM_folder' in tuning_conf:
        SC.tuning.RM_folder = tuning_conf['model_RM_folder']

    if 'multipoles' in tuning_conf:
        SC.tuning.multipoles = resolve_controls(SC, "multipoles", tuning_conf["multipoles"])

    if 'bba_magnets' in tuning_conf:
        SC.tuning.bba_magnets = resolve_controls(SC, "bba_magnets", tuning_conf["bba_magnets"])

    if 'tune' in tuning_conf:
        tune_conf = tuning_conf['tune']
        assert 'controls_1' in tune_conf, 'controls_1 missing from tune configuration.'
        assert 'controls_2' in tune_conf, 'controls_2 missing from tune configuration.'

        SC.tuning.tune.controls_1 = resolve_controls(SC, "tune/controls_1", tune_conf["controls_1"])
        SC.tuning.tune.controls_2 = resolve_controls(SC, "tune/controls_2", tune_conf["controls_2"])

    if 'c_minus' in tuning_conf:
        c_minus_conf = tuning_conf['c_minus']
        if 'controls' in c_minus_conf:
            SC.tuning.c_minus.controls = resolve_controls(SC, "c_minus/controls", c_minus_conf["controls"])
