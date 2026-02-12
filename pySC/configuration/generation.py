from typing import Optional
import logging

from ..core.lattice import ATLattice
from ..core.simulated_commissioning import SimulatedCommissioning
from .load_config import load_yaml
from .magnets_conf import configure_magnets
from .bpm_system_conf import configure_bpms
from .rf_conf import configure_rf
from .supports_conf import configure_supports
from .tuning_conf import configure_tuning
from .injection_conf import configure_injection
from .general import scale_error_table

logger = logging.getLogger(__name__)

def generate_SC(yaml_filepath: str, seed: int = 1, scale_errors: Optional[int] = None, sigma_truncate: Optional[int] = None) -> SimulatedCommissioning:
    config_dict = load_yaml(yaml_filepath)

    assert 'lattice' in config_dict, f"'lattice' is missing in the configuration file: {yaml_filepath}."

    if 'error_table' not in config_dict.keys():
        logger.warning(f"'error_table' is missing in the configuration file: {yaml_filepath}. It will be empty!")
    else:
        if scale_errors is not None:
            print(f"Scaling error_table by {scale_errors}")
            config_dict['error_table'] = scale_error_table(config_dict['error_table'], scale=scale_errors)

    if 'magnets' not in config_dict.keys():
        logger.warning("'magnets' is missing in the configuration file. Generating an empty one.")

    if 'bpms' not in config_dict.keys():
        logger.warning("'bpms' is missing in the configuration file. Generating an empty one.")

    # TODO: maybe put in a separate module
    if config_dict['lattice']['simulator'] == 'at':
        lattice_file = config_dict['lattice']['lattice_file']

        if 'no_6d' in config_dict['lattice']:
            no_6d = config_dict['lattice']['no_6d']
        else:
            no_6d = False

        if 'use' in config_dict['lattice']:
            use = config_dict['lattice']['use']
        else:
            use = 'RING'

        if 'naming' in config_dict['lattice']:
            naming = config_dict['lattice']['naming']
        else:
            naming = None

        logger.info(f'Loading AT lattice from {lattice_file}')
        lattice = ATLattice(lattice_file=lattice_file, no_6d=no_6d, use=use, naming=naming)
    else:
        raise NotImplementedError(f"Simulator {config_dict['lattice']['simulator']} is not implemented.")

    # Create the SimulatedCommissioning instance
    SC = SimulatedCommissioning(lattice=lattice, configuration=config_dict, seed=seed)
    SC.rng.default_truncation = sigma_truncate

    logger.info('Configuring magnets...')
    configure_magnets(SC)

    # initialize magnets

    logger.info('Configuring BPMs...')
    configure_bpms(SC)

    logger.info('Configuring rf...')
    configure_rf(SC)

    logger.info('Configuring supports...')
    configure_supports(SC)

    logger.info('Configuring tuning...')
    configure_tuning(SC)

    logger.info('Configuring injection...')
    configure_injection(SC)
    return SC