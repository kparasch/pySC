from typing import Any
import logging
import numpy as np

from ..core.simulated_commissioning import SimulatedCommissioning
from ..core.lattice import ATLattice
from ..core.control import LinearConv
from ..core.magnet import ControlMagnetLink, MAGNET_NAME_TYPE
from ..core.multipolar_imperfections import ImperfectionsModelFactory
from .general import get_error, get_indices_and_names
from .supports_conf import generate_element_misalignments

logger = logging.getLogger(__name__)

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

    is_shifted = magnet_category_conf.get('shifted', False)
    magnet_length = SC.lattice.get_length(index)
    if is_shifted and not SC.lattice.is_dipole(index):
        raise ValueError(
            f"magnets/{magnet_category_name}: shifted=true expects a dipole "
            f"at index {index} ({magnet_name})."
        )

    if 'components' in magnet_category_conf:
        components = []
        cal_errors = []
        for comp_dict in magnet_category_conf['components']:
            component, cal_error = comp_dict.copy().popitem()
            components.append(component)
            cal_errors.append(cal_error)

        magnet_settings.add_individually_powered_magnet(
            sim_index=index, 
            controlled_components=components,
            magnet_name=magnet_name, 
            magnet_length=magnet_length,
            to_design=to_design
        )

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
            if type(SC.lattice) is ATLattice and component == 'B1' and SC.lattice.is_dipole(index):
                # in AT when we have a dipole with bending angle it is a special case,
                # setpoint points to bending angle, but B1 multipole (PolynomB[0]) should be changed
                # TODO: maybe put this on the side of ATLattice?? 
                #   when we do get we return BendingAngle/length + PolynomB[0],
                #   when we do set we subtract BendingAngle/length from setpoint and then set it to PolynomB[0]
                # in XSuite, h controls the reference frame and k0 the strength of the dipole. We can just act on k0.
                bending_angle = SC.lattice.get_bending_angle(index)
                offset = - bending_angle / magnet_length
                setpoint = bending_angle / magnet_length
            else:
                #otherwise it is just the multipole
                offset = 0
                setpoint = SC.lattice.get_magnet_component(index, component_type=component_type, order=order)
                if component[-1] == 'L':
                    length = magnet_length
                    setpoint = setpoint * length

            if component in components_to_invert:
                factor *= -1
                components_to_invert.remove(component) 

            magnet_settings.controls[control_name].setpoint = setpoint
            magnet_settings.links[link_name].error.factor = factor
            magnet_settings.links[link_name].error.offset = offset

        if is_shifted:
            quad_components = [component for component in components if component in ('B2', 'B2L')]
            if len(quad_components) != 1:
                raise ValueError(f"magnets/{magnet_category_name}: shifted=true requires B2 or B2L in control")

            component = quad_components[0]
            control_name = f'{magnet_name}/{component}'
            source_link = magnet_settings.links[f'{control_name}->{control_name}']
            k1 = SC.lattice.get_magnet_component(index, component_type='B', order=1)
            if k1 == 0:
                raise ValueError(f"magnets/{magnet_category_name}: shifted=true requires a non-zero quadrupole component at index {index} ({magnet_name}).")

            k0 = SC.lattice.get_magnet_component(index, component_type='B', order=0)
            reference_k0 = 0.0
            if type(SC.lattice) is ATLattice:
                reference_k0 = SC.lattice.get_bending_angle(index) / magnet_length
                k0 += reference_k0
            design_shift = k0 / k1

            scale = magnet_length if source_link.is_integrated else 1.0
            factor = design_shift * source_link.error.factor / scale
            offset = design_shift * source_link.error.offset / scale - reference_k0
            target_name = f'{magnet_name}/B1'
            magnet_settings.add_link(ControlMagnetLink(
                link_name=f'{control_name}->{target_name}',
                magnet_name=magnet_name,
                control_name=control_name,
                component='B',
                order=1,
                error=LinearConv(factor=factor, offset=offset),
                is_integrated=False,
            ))
            magnet_settings.magnets[magnet_name].offset_B[0] = 0.0

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

def generate_magnet_in_series_controls(SC: SimulatedCommissioning, indices: list[int], magnet_names: list[MAGNET_NAME_TYPE],
                                    magnet_category_conf: dict[str, Any], magnet_category_name: str, to_design: bool = False) -> list[str]:
    error_table = dict.get(SC.configuration, 'error_table', {}) # defaults to empty error_table if not declared
    new_control_list = []

    if to_design:
        magnet_settings = SC.design_magnet_settings
    else:
        magnet_settings = SC.magnet_settings

    if not indices: #indices is empty
        return []

    components_to_invert = dict.get(magnet_category_conf, 'invert', []).copy() # defaults to empty list if not declared
    # we need to copy because we remove elements later to check for undeclared components to invert

    is_shifted = magnet_category_conf.get('shifted', False)
    magnet_lengths = [SC.lattice.get_length(index) for index in indices]
    if is_shifted:
        raise NotImplementedError("Shifted (by design) magnets in series is not implemented yet.")

    if 'components' in magnet_category_conf:
        components = []
        cal_errors = []
        for comp_dict in magnet_category_conf['components']:
            component, cal_error = comp_dict.copy().popitem()
            components.append(component)
            cal_errors.append(cal_error)

        link_names_by_component = magnet_settings.add_magnets_in_series(
            sim_indices=indices,
            controlled_components=components,
            category_name=magnet_category_name,
            magnet_names=magnet_names,
            magnet_lengths=magnet_lengths,
            to_design=to_design,
        )

        for component, cal_error in zip(components, cal_errors):
            control_name = f'{magnet_category_name}/{component}'
            link_names = link_names_by_component[component]
            new_control_list.append(control_name)
            setpoints = []
            for link_name in link_names:
                magnet_name = magnet_settings.links[link_name].magnet_name
                index = magnet_settings.magnets[magnet_name].sim_index
                magnet_length = magnet_settings.magnets[magnet_name].length
                if to_design:
                    factor = 1
                else:
                    sig = get_error(cal_error, error_table)
                    factor = SC.rng.normal_trunc(1, sig)

                component_type, order = magnet_settings.validate_one_component(component)
                if type(SC.lattice) is ATLattice and component == 'B1' and SC.lattice.is_dipole(index):
                    # in AT when we have a dipole with bending angle it is a special case,
                    # setpoint points to bending angle, but B1 multipole (PolynomB[0]) should be changed
                    # TODO: maybe put this on the side of ATLattice?? 
                    #   when we do get we return BendingAngle/length + PolynomB[0],
                    #   when we do set we subtract BendingAngle/length from setpoint and then set it to PolynomB[0]
                    # in XSuite, h controls the reference frame and k0 the strength of the dipole. We can just act on k0.
                    bending_angle = SC.lattice.get_bending_angle(index)
                    offset = - bending_angle / magnet_length
                    setpoint = bending_angle / magnet_length
                else:
                    #otherwise it is just the multipole
                    offset = 0
                    setpoint = SC.lattice.get_magnet_component(index, component_type=component_type, order=order)
                    if component[-1] == 'L':
                        length = magnet_length
                        setpoint = setpoint * length

                if component in components_to_invert:
                    factor *= -1

                setpoints.append(setpoint)

                magnet_settings.links[link_name].error.factor = factor
                magnet_settings.links[link_name].error.offset = offset

            if component in components_to_invert:
                components_to_invert.remove(component) 

            same = np.allclose(setpoints, setpoints[0], rtol=1e-9, atol=1e-12)
            if not same:
                logger.warning(f"Setpoints in series of magnets '{magnet_category_name}' are not all the same for component '{component}'."
                               f"Mean setpoint = {np.mean(setpoints):.3e}, St.dev. setpoint = {np.std(setpoints):.3e}."
                               "Setting the mean.")

            mean_setpoint = np.mean(setpoints)
            magnet_settings.controls[control_name].setpoint = mean_setpoint

    assert len(components_to_invert) == 0, f"Found undeclared components in components to invert: magnets/{magnet_category_name}/invert: {components_to_invert}."

    parameters_table = dict.get(SC.configuration, 'parameters', {}) # defaults to empty error_table if not declared
    if 'limits' in magnet_category_conf:
        for comp_dict in magnet_category_conf['limits']:
            component, limit_name = comp_dict.copy().popitem()
            if limit_name not in parameters_table:
                raise Exception(f'ERROR: limits {limit_name} were not found in error_table.')
            limit = float(parameters_table[limit_name])
            control_name = f'{magnet_category_name}/{component}'
            if control_name not in new_control_list:
                raise Exception('ERROR: Invalid limit.') ## TODO make more verbose
            magnet_settings.controls[control_name].limits = (-abs(limit), abs(limit))

    return new_control_list

def configure_magnets(SC: SimulatedCommissioning):
    # get magnets configuration, return empty dict if not there
    magnet_conf = dict.get(SC.configuration, 'magnets', {})

    if 'multipolar_imperfection_models' in SC.configuration:
        multipolar_imperfection_models = dict.get(SC.configuration, 'multipolar_imperfection_models')
        imperfection_model_factories: dict[str, ImperfectionsModelFactory] = {}
        for model_name, model_object in multipolar_imperfection_models.items():
            imperfection_model_factories[model_name] = ImperfectionsModelFactory.model_validate({'factories': model_object})

    for magnet_category_name in magnet_conf.keys():
        magnet_category_conf = magnet_conf[magnet_category_name]
        magnet_list = []
        control_list = []

        indices, magnet_names = get_indices_and_names(SC, magnet_category_name, magnet_category_conf)

        is_series = magnet_category_conf.get("series", False)
        if is_series:
            magnet_list = magnet_names
            for index, magnet_name in zip(indices, magnet_names):
                # misalignments
                generate_element_misalignments(SC, index, magnet_category_conf)
            new_controls = generate_magnet_in_series_controls(SC, indices, magnet_names, magnet_category_conf, magnet_category_name=magnet_category_name)
            _ = generate_magnet_in_series_controls(SC, indices, magnet_names, magnet_category_conf, magnet_category_name=magnet_category_name, to_design=True)
            control_list = control_list + new_controls
        else:
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

        if 'imperfections' in magnet_category_conf:
            if 'multipolar_imperfection_models' not in SC.configuration:
                raise Exception("'multipolar_imperfection_models' not found in configuration.")

            imperfections_model_name = magnet_category_conf['imperfections']
            for magnet_name in magnet_names:
                if SC.magnet_settings.magnets[magnet_name].imperfections is not None:
                    raise Exception(f"Magnet {magnet_name} of {magnet_category_name} already has an imperfections model.")
                if imperfections_model_name not in imperfection_model_factories:
                    raise ValueError(f"{imperfections_model_name} not found in the multipolar imperfection models.")
                factory = imperfection_model_factories[imperfections_model_name]
                SC.magnet_settings.magnets[magnet_name].imperfections = factory.create(SC.rng)

                required_max_order = SC.magnet_settings.magnets[magnet_name].imperfections.max_order
                SC.magnet_settings.magnets[magnet_name].ensure_max_order(required_max_order=required_max_order)

    SC.magnet_settings.connect_links()
    SC.magnet_settings.sendall()
    SC.design_magnet_settings.connect_links()
    SC.design_magnet_settings.sendall()
