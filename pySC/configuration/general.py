from typing import Optional, Any
from ..core.new_simulated_commissioning import SimulatedCommissioning
from ..core.magnet import MAGNET_NAME_TYPE

def get_error(error_name: Optional[str], error_table: dict) -> float:
    if error_name is None:
        return 0
    if error_name not in error_table:
        raise Exception(f"Error: '{error_name}' not found in error_table.")
    return float(error_table[error_name])

def scale_error_table(error_table: dict, scale: float = 1) -> dict:
    for key in error_table.keys():
        error = get_error(error_name=key, error_table=error_table)
        error_table[key] = str(scale * error)
    return error_table

def get_indices_with_regex(SC: SimulatedCommissioning, category_name: str, category_conf: dict[str, Any]) -> list[int]:
    indices = SC.lattice.find_with_regex(category_conf['regex'])

    if 'exclude' in category_conf:
        exclude_indices = SC.lattice.find_with_regex(category_conf['exclude']) 
        indices = list(set(indices) - set(exclude_indices))
    print(f"Found {len(indices)} ({category_name}) matching regex '{category_conf['regex']}'")
    return indices

def get_indices_and_names(SC: SimulatedCommissioning, category_name: str, category_conf: dict[str, Any]) -> tuple[list[int], list[MAGNET_NAME_TYPE]]:
    if 'regex' in category_conf:
        indices = get_indices_with_regex(SC, category_name, category_conf)
        names = list(map(str, indices))
    else:
        raise NotImplementedError("Only regex search is implemented.")

    return indices, names