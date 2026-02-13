import numpy as np
import logging

from ..core.simulated_commissioning import SimulatedCommissioning
from ..core.bpm_system import BPM_FIELDS_TO_INITIALISE
from .general import get_error, get_indices_and_names
from .supports_conf import generate_element_misalignments

logger = logging.getLogger(__name__)

def configure_bpms(SC: SimulatedCommissioning) -> None:
    # get magnets configuration, return empty dict if not there
    bpms_conf = dict.get(SC.configuration, 'bpms', {})

    error_table = dict.get(SC.configuration, 'error_table', {}) # defaults to empty error_table if not declared

    bpms_indices = []
    bpms_names = []
    bpms_orbit_noise = []
    bpms_tbt_noise = []
    bpms_calibration_error_x = []
    bpms_calibration_error_y = []
    bpms_categories = []

    if len(bpms_conf.keys()) > 1:
        logger.fatal('More than one bpm category found in the configuration file. Not tested! Proceed with caution and ask for help!')

    for bpms_category in bpms_conf.keys():
        bpms_category_conf = bpms_conf[bpms_category]

        indices, names = get_indices_and_names(SC, bpms_category, bpms_category_conf)

        # make sure indices are not already in bpms_indices
        # if they are, raise an error.
        if not len(set(bpms_indices).intersection(set(indices))) == 0:
            logger.fatal(f'At least one bpm in category {bpms_category} has already been registered.')
            raise Exception(f'ERROR: At least one bpm in category {bpms_category} has already been registered.')
        bpms_indices = bpms_indices + indices
        bpms_names = bpms_names + names
        nbpm = len(bpms_indices)

        bpms_categories = bpms_categories + [bpms_category] * nbpm


        if 'orbit_noise' in bpms_category_conf:
            orbit_noise = get_error(bpms_category_conf['orbit_noise'], error_table)
        else:
            logger.warning(f'No orbit_noise was found for bpms: {bpms_category}.')
            orbit_noise = 0
        bpms_orbit_noise = bpms_orbit_noise + [orbit_noise] * nbpm

        if 'tbt_noise' in bpms_category_conf:
            tbt_noise = get_error(bpms_category_conf['tbt_noise'], error_table)
        else:
            logger.warning(f'No tbt_noise was found for bpms: {bpms_category}.')
            tbt_noise = 0
        bpms_tbt_noise = bpms_tbt_noise + [tbt_noise] * nbpm

        if 'calibration_error' in bpms_category_conf:
            sig = get_error(bpms_category_conf['calibration_error'], error_table)
        else:
            logger.warning(f'No calibration_error was found for bpms: {bpms_category}.')
            sig = 0

        for _ in range(len(indices)):
            # during application of the errors, the calibration error is recented around 1
            # so we don't add it here. 
            bpms_calibration_error_x.append(SC.rng.normal_trunc(0, sig))
            bpms_calibration_error_y.append(SC.rng.normal_trunc(0, sig))

    # sort everything to make sure indices are in order
    argsort = np.argsort(bpms_indices).tolist()
    bpms_indices = [bpms_indices[i] for i in argsort]
    bpms_names = [bpms_names[i] for i in argsort]
    bpms_orbit_noise = [bpms_orbit_noise[i] for i in argsort]
    bpms_tbt_noise = [bpms_tbt_noise[i] for i in argsort]
    bpms_calibration_error_x = [bpms_calibration_error_x[i] for i in argsort]
    bpms_calibration_error_y = [bpms_calibration_error_y[i] for i in argsort]
    bpms_categories = [bpms_categories[i] for i in argsort]

    SC.bpm_system.indices = bpms_indices
    SC.bpm_system.names = bpms_names
    SC.bpm_system.calibration_errors_x = np.array(bpms_calibration_error_x)
    SC.bpm_system.calibration_errors_y = np.array(bpms_calibration_error_y)
    SC.bpm_system.noise_co_x = np.array(bpms_orbit_noise)
    SC.bpm_system.noise_co_y = np.array(bpms_orbit_noise)
    SC.bpm_system.noise_tbt_x = np.array(bpms_tbt_noise)
    SC.bpm_system.noise_tbt_y = np.array(bpms_tbt_noise)

    nbpm = len(bpms_indices)
    for field in BPM_FIELDS_TO_INITIALISE:
        setattr(SC.bpm_system, field, np.zeros(nbpm, dtype=float))

    for index, bpm_category in zip(bpms_indices, bpms_categories):
        generate_element_misalignments(SC, index, bpms_conf[bpm_category])
    SC.bpm_system.update_rot_matrices()
