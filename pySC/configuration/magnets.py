from typing import Any
from ..core.new_simulated_commissioning import SimulatedCommissioning
from ..core.magnet import MAGNET_NAME_TYPE
from .general import get_error, get_indices_and_names
from .supports import generate_element_misalignments

def generate_default_magnet_control(SC: SimulatedCommissioning, index: int, magnet_name: MAGNET_NAME_TYPE, magnet_category_conf: dict[str, Any]) -> list[str]:
    error_table = dict.get(SC.configuration, 'error_table', {}) # defaults to empty error_table if not declared
    new_control_list = []

    if 'components' in magnet_category_conf:
        components = []
        cal_errors = []
        for comp_dict in magnet_category_conf['components']:
            component, cal_error = comp_dict.copy().popitem()
            components.append(component)
            cal_errors.append(cal_error)

        SC.magnet_settings.add_individually_powered_magnet(
            sim_index=index, controlled_components=components,
            magnet_name=magnet_name)

        for component, cal_error in zip(components, cal_errors):
            control_name = f'{magnet_name}/{component}'
            link_name = f'{control_name}->{control_name}'

            new_control_list.append(control_name)
            sig = get_error(cal_error, error_table)
            factor = SC.rng.normal_trunc(1, sig)

            component_type, order = SC.magnet_settings.validate_one_component(component)
            if component == 'B1' and SC.lattice.is_dipole(index):
                # when we have a dipole with bending angle it is a special case,
                # setpoint points to bending angle, but B1 multipole (PolynomB[0]) should be changed
                bending_angle = SC.lattice.get_bending_angle(index)
                magnet_length = SC.lattice.get_length(index)
                offset = - bending_angle / magnet_length
                setpoint = bending_angle / magnet_length
            else:
                #otherwise it is just the multipole
                offset = 0
                setpoint = SC.lattice.get_magnet_component(index, component_type=component_type, order=order)

            SC.magnet_settings.controls[control_name].setpoint = setpoint
            SC.magnet_settings.links[link_name].error.factor = factor
            SC.magnet_settings.links[link_name].error.offset = offset
    return new_control_list


def configure_magnets(SC: SimulatedCommissioning):
    # get magnets configuration, return empty dict if not there
    magnet_conf = dict.get(SC.configuration, 'magnets', {})

    for magnet_category in magnet_conf.keys():
        magnet_category_conf = magnet_conf[magnet_category]
        magnet_list = []
        control_list = []

        indices, magnet_names = get_indices_and_names(SC, magnet_category, magnet_category_conf)

        for index, magnet_name in zip(indices, magnet_names):
            magnet_list.append(magnet_name)
            # misalignments
            generate_element_misalignments(SC, index, magnet_category_conf)
            # calibration errors
            new_controls = generate_default_magnet_control(SC, index, magnet_name, magnet_category_conf)
            control_list = control_list + new_controls
        SC.magnet_arrays[magnet_category] = magnet_list
        SC.control_arrays[magnet_category] = control_list