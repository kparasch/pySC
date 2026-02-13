from typing import Any
from ..core.simulated_commissioning import SimulatedCommissioning
from ..core.magnet import MAGNET_NAME_TYPE
from .general import get_error, get_indices_and_names
from .supports_conf import generate_element_misalignments

def generate_default_magnet_control(SC: SimulatedCommissioning, index: int, magnet_name: MAGNET_NAME_TYPE,
                                    magnet_category_conf: dict[str, Any], magnet_category_name: str, to_design: bool = False) -> list[str]:
    error_table = dict.get(SC.configuration, 'error_table', {}) # defaults to empty error_table if not declared
    new_control_list = []

    if to_design:
        magnet_settings = SC.design_magnet_settings
    else:
        magnet_settings = SC.magnet_settings

    components_to_invert = dict.get(magnet_category_conf, 'invert', []).copy() # defaults to empty list if not declared
    # we need to copy because we remove elements later to check for undeclared components to invert

    if 'components' in magnet_category_conf:
        components = []
        cal_errors = []
        for comp_dict in magnet_category_conf['components']:
            component, cal_error = comp_dict.copy().popitem()
            components.append(component)
            cal_errors.append(cal_error)

        magnet_length = SC.lattice.get_length(index)
        magnet_settings.add_individually_powered_magnet(
            sim_index=index, controlled_components=components,
            magnet_name=magnet_name, magnet_length=magnet_length,
            to_design=to_design)

        for component, cal_error in zip(components, cal_errors):
            control_name = f'{magnet_name}/{component}'
            link_name = f'{control_name}->{control_name}'

            new_control_list.append(control_name)
            if to_design:
                factor = 1
            else:
                sig = get_error(cal_error, error_table)
                factor = SC.rng.normal_trunc(1, sig)

            component_type, order = magnet_settings.validate_one_component(component)
            if component == 'B1' and SC.lattice.is_dipole(index):
                # when we have a dipole with bending angle it is a special case,
                # setpoint points to bending angle, but B1 multipole (PolynomB[0]) should be changed
                bending_angle = SC.lattice.get_bending_angle(index)
                offset = - bending_angle / magnet_length
                setpoint = bending_angle / magnet_length
            else:
                #otherwise it is just the multipole
                offset = 0
                setpoint = SC.lattice.get_magnet_component(index, component_type=component_type, order=order)
                if component[-1] == 'L':
                    length = SC.lattice.get_length(index)
                    setpoint = setpoint * length

            if component in components_to_invert:
                factor *= -1
                components_to_invert.remove(component) 

            magnet_settings.controls[control_name].setpoint = setpoint
            magnet_settings.links[link_name].error.factor = factor
            magnet_settings.links[link_name].error.offset = offset

    assert len(components_to_invert) == 0, f"Found undeclared components in components to invert: magnets/{magnet_category_name}/invert: {components_to_invert}."


    parameters_table = dict.get(SC.configuration, 'parameters', {}) # defaults to empty error_table if not declared
    if 'limits' in magnet_category_conf:
        for comp_dict in magnet_category_conf['limits']:
            component, limit_name = comp_dict.copy().popitem()
            if limit_name not in parameters_table:
                raise Exception(f'ERROR: limits {limit_name} were not found in error_table.')
            limit = float(parameters_table[limit_name])
            control_name = f'{magnet_name}/{component}'
            if control_name not in new_control_list:
                raise Exception('ERROR: Invalid limit.') ## TODO make more verbose
            magnet_settings.controls[control_name].limits = (-abs(limit), abs(limit))

    return new_control_list


def configure_magnets(SC: SimulatedCommissioning):
    # get magnets configuration, return empty dict if not there
    magnet_conf = dict.get(SC.configuration, 'magnets', {})

    for magnet_category_name in magnet_conf.keys():
        magnet_category_conf = magnet_conf[magnet_category_name]
        magnet_list = []
        control_list = []

        indices, magnet_names = get_indices_and_names(SC, magnet_category_name, magnet_category_conf)

        for index, magnet_name in zip(indices, magnet_names):
            magnet_list.append(magnet_name)
            # misalignments
            generate_element_misalignments(SC, index, magnet_category_conf)
            # calibration errors
            new_controls = generate_default_magnet_control(SC, index, magnet_name, magnet_category_conf, magnet_category_name=magnet_category_name)
            _ = generate_default_magnet_control(SC, index, magnet_name, magnet_category_conf, magnet_category_name=magnet_category_name, to_design=True)
            control_list = control_list + new_controls
        SC.magnet_arrays[magnet_category_name] = magnet_list
        SC.control_arrays[magnet_category_name] = control_list

    SC.magnet_settings.connect_links()
    SC.magnet_settings.sendall()
    SC.design_magnet_settings.connect_links()
    SC.design_magnet_settings.sendall()