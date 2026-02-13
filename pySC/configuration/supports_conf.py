
from typing import  Any
from ..core.simulated_commissioning import SimulatedCommissioning
from .general import get_error, get_indices_and_names
import logging

logger = logging.getLogger(__name__)

SQRT2 = 2**0.5
ZERO_LENGTH_THRESHOLD = 1e-6

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

        alignment = dict.get(level_conf, 'alignment', 'absolute') # defaults to absolute if not specified
        if alignment not in ['absolute', 'relative']:
            raise Exception('Unknown alignment mode: {alignment}. Only "absolute" and "relative" are supported.')

        zero_length_supports = []
        for index_start, index_end in zip(indices_start, indices_end):
            # If in the future I want to give names to supports (e.g. girders),
            # here I can have it pass the name from get_indices_and_names (TODO)
            support_index = SC.support_system.add_support(index_start, index_end, name=level_name, level=level)

            this_support = SC.support_system.data[f'L{level}'][support_index]

            support_has_zero_length = False
            if this_support.length < ZERO_LENGTH_THRESHOLD:
                support_has_zero_length = True
                zero_length_supports.append(support_index)

            if 'dx' in level_conf:
                sigma = get_error(level_conf['dx'], error_table)
                if alignment == 'relative':
                    sigma = sigma / SQRT2
                this_support.start.dx = SC.rng.normal_trunc(0, sigma)
                if support_has_zero_length:
                    this_support.end.dx = this_support.start.dx
                else:
                    this_support.end.dx = SC.rng.normal_trunc(0, sigma)

            if 'dy' in level_conf:
                sigma = get_error(level_conf['dy'], error_table)
                if alignment == 'relative':
                    sigma = sigma / SQRT2
                this_support.start.dy = SC.rng.normal_trunc(0, sigma)
                if support_has_zero_length:
                    this_support.end.dy = this_support.start.dy
                else:
                    this_support.end.dy = SC.rng.normal_trunc(0, sigma)

            if 'roll' in level_conf:
                sigma = get_error(level_conf['roll'], error_table)
                this_support.roll = SC.rng.normal_trunc(0, sigma)
        if len(zero_length_supports):
            logger.warning(f'Found {len(zero_length_supports)} zero-length supports in level {level} ({category_name}).')

    SC.support_system.resolve_graph()
    SC.support_system.update_all()