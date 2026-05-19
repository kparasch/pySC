"""Measure beam-based alignment using an operational interface."""

import json
import logging

from interface import DATA_FOLDER, Interface
from pySC import measure_bba
from pySC.apps.bba import BBACode, BBAAnalysis


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

interface = Interface()

with open("bba_config.json") as fp:
    bba_config = json.load(fp)

bpm_name = next(iter(bba_config))
config = bba_config[bpm_name]

generator = measure_bba(
    interface=interface,
    bpm_name=bpm_name,
    config=config,
    shots_per_orbit=2,
    n_corr_steps=7,
    bipolar=True,
    skip_save=False,
    folder_to_save=DATA_FOLDER,
)

for code, measurement in generator:
    if code is BBACode.HORIZONTAL_DONE:
        result = BBAAnalysis.analyze(measurement.H_data)
        logger.info("BBA offset H = %.1f +- %.1f um", result.offset * 1e6, result.offset_error * 1e6)

    if code is BBACode.VERTICAL_DONE:
        result = BBAAnalysis.analyze(measurement.V_data)
        logger.info("BBA offset V = %.1f +- %.1f um", result.offset * 1e6, result.offset_error * 1e6)
