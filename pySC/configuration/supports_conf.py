
from typing import  Any
from ..core.new_simulated_commissioning import SimulatedCommissioning
from .general import get_error, get_indices_and_names

SQRT2 = 2**0.5

def generate_element_misalignments(SC: SimulatedCommissioning, index: int, category_conf: dict[str, Any]) -> None:
    error_table = dict.get(SC.configuration, 'error_table', {}) # defaults to empty error_table if not declared
    SC.support_system.add_element(index)
    for error_type in ['dx', 'dy', 'dz', 'roll', 'yaw', 'pitch']:
        if error_type in category_conf:
            sigma = get_error(category_conf[error_type], error_table)
            setattr(SC.support_system.data['L0'][index], error_type, SC.rng.normal_trunc(0, sigma))

def configure_supports(SC: SimulatedCommissioning):
    # get support configuration, return empty dict if not there
    supports_conf = dict.get(SC.configuration, 'supports', {})
    error_table = dict.get(SC.configuration, 'error_table', {}) # defaults to empty error_table if not declared

    for level_conf in supports_conf:
        level = level_conf['level']

        if 'name' in level_conf['name']:
            level_name = level_conf['name'] 
            category_name = level_name
        else:
            level_name = None
            category_name = f'L{level}'

        indices_start, _ = get_indices_and_names(SC, f'{category_name}/start_endpoints', level_conf['start_endpoints'])
        indices_end, _ = get_indices_and_names(SC, f'{category_name}/end_endpoints', level_conf['end_endpoints'])

        if len(indices_start) != len(indices_end):
            raise Exception(f'Unequal number of endpoints found in support level {level} ({category_name}).')

        alignment = dict.get(level_conf, 'alignment', 'absolute')
        if alignment not in ['absolute', 'relative']:
            raise Exception('Unknown alignment mode: {alignment}. Only "absolute" and "relative" are supported.')

        for index_start, index_end in zip(indices_start, indices_end):
            # If in the future I want to give names to supports (e.g. girders),
            # here I can have it pass the name from get_indices_and_names (TODO)
            support_index = SC.support_system.add_support(index_start, index_end, name=level_name, level=level)

            if 'dx' in level_conf:
                sigma = get_error(level_conf['dx'], error_table)
                if alignment == 'relative':
                    sigma = sigma / SQRT2
                SC.support_system.data[f'L{level}'][support_index].start.dx = SC.rng.normal_trunc(0, sigma)

            if 'dy' in level_conf:
                sigma = get_error(level_conf['dy'], error_table)
                if alignment == 'relative':
                    sigma = sigma / SQRT2
                SC.support_system.data[f'L{level}'][support_index].start.dy = SC.rng.normal_trunc(0, sigma)

            if 'roll' in level_conf:
                sigma = get_error(level_conf['roll'], error_table)
                SC.support_system.data[f'L{level}'][support_index].roll = SC.rng.normal_trunc(0, sigma)