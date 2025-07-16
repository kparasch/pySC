from typing import Optional
from ..core.lattice import ATLattice
from ..core.new_simulated_commissioning import SimulatedCommissioning
from .load_config import load_yaml
from .magnets import configure_magnets
from .bpms import configure_bpms
from .rf import configure_rf
from .supports import configure_supports
from .general import scale_error_table


def generate_SC(yaml_filepath: str, seed: int = 1, scale_errors: Optional[int] = None) -> SimulatedCommissioning:
    config_dict = load_yaml(yaml_filepath)

    assert 'lattice' in config_dict, f"'lattice' is missing in the configuration file: {yaml_filepath}."

    if 'error_table' not in config_dict.keys():
        print(f"WARNING: 'error_table' is missing in the configuration file: {yaml_filepath}. It will be empty!")
    else:
        if scale_errors is not None:
            print(f"Scaling error_table by {scale_errors}")
            config_dict['error_table'] = scale_error_table(config_dict['error_table'], scale=scale_errors)

    if 'magnets' not in config_dict.keys():
        print("WARNING: 'magnets' is missing in the configuration file. Generating an empty one.")

    if 'bpms' not in config_dict.keys():
        print("WARNING: 'bpms' is missing in the configuration file. Generating an empty one.")

    if config_dict['lattice']['simulator'] == 'at':
        lattice_file = config_dict['lattice']['lattice_file']
        print(f'Loading AT lattice from {lattice_file}')
        lattice = ATLattice(lattice_file=lattice_file)
    else:
        raise NotImplementedError(f"Simulator {config_dict['lattice']['simulator']} is not implemented.")

    # Create the SimulatedCommissioning instance
    SC = SimulatedCommissioning(lattice=lattice, configuration=config_dict, seed=seed)

    print('Configuring magnets...')
    configure_magnets(SC)

    # initialize magnets
    SC.magnet_settings.connect_links()
    SC.magnet_settings.sendall()

    print('Configuring BPMs...')
    configure_bpms(SC)

    print('Configuring rf...')
    configure_rf(SC)

    print('Configuring supports...')
    configure_supports(SC)
    SC.support_system.update_all()

    return SC