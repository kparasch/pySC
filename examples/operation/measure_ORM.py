"""Measure an orbit response matrix using an operational interface."""

import json
import logging

from interface import DATA_FOLDER, Interface
from pySC import ResponseMatrix, measure_ORM, orbit_correction
from pySC.apps.codes import ResponseCode


START_MEASUREMENT = False
DELTA = 100e-6

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

interface = Interface()

response_matrix = ResponseMatrix.from_json("ideal_orm.json")
with open("name_mapping.json") as fp:
    name_mapping = json.load(fp)
response_matrix.input_names = [name_mapping[name] for name in response_matrix.input_names]

corrector_names = response_matrix.input_names

generator = measure_ORM(
    interface=interface,
    corrector_names=corrector_names,
    delta=DELTA,
    shots_per_orbit=1,
    bipolar=True,
    skip_save=False,
    folder_to_save=DATA_FOLDER,
)

if START_MEASUREMENT:
    for code, measurement in generator:
        logger.info(
            "%s/%s, code=%s, last_corrector=%s",
            measurement.last_number + 1,
            len(corrector_names),
            code.name,
            measurement.last_input,
        )

        if code is ResponseCode.MEASURING:
            last_corrector = measurement.last_input
            response_matrix.enable_all_inputs()
            response_matrix.disable_all_inputs_but([last_corrector])
            trims = orbit_correction(
                interface=interface,
                response_matrix=response_matrix,
                method="micado",
                parameter=1,
                apply=True,
            )
            logger.info("Corrected orbit with: %s", trims)
