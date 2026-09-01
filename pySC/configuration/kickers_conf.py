import logging
from typing import Literal

from .general import pySCConfigurationError
from .tuning_conf import configure_family, sort_controls
from ..core.simulated_commissioning import SimulatedCommissioning
from ..core.kickers import SingleKickProgram, ACProgram, WhiteNoiseProgram

KICK_PROGRAM_MANDATORY_FIELDS = ['control', 'amplitude']
ALLOWED_PROGRAMS = ["single_kick", "ac", "white_noise"]

logger = logging.getLogger(__name__)

def get_kicker_control(SC: SimulatedCommissioning, program_name: str, program_conf: dict) -> str:
    controls = configure_family(SC, config_dict=program_conf['control'])
    if len(controls) > 1:
        sorted_controls = sort_controls(SC, controls)
        control = sorted_controls[0]
        logger.warning(f"More than one control found for kickers/{program_name}. Will use the first one.")
    elif not len(controls):
        raise pySCConfigurationError(f"No controls were found for kickers/{program_name}.")
    else:
        control = controls[0]
    return control

def get_kicker_program_arguments(program_conf: dict, program_name: str, program_kind: Literal["single_kick", "ac", "white_noise"]):
    if program_kind == "single_kick":
        allowed = set(SingleKickProgram.model_fields)
    elif program_kind == "ac":
        allowed = set(ACProgram.model_fields)
    elif program_kind == "white_noise":
        allowed = set(WhiteNoiseProgram.model_fields)
    else:
        raise pySCConfigurationError(f"Unknown kicker program kind {program_kind}.")

    allowed.discard("kind")
    allowed.discard("control")

    args = {}
    for field, value in program_conf.items():
        if field == "control":
            continue
        if field not in allowed:
            raise pySCConfigurationError(f"Unknown field kickers/{program_kind}/{program_name}/{field}.")
        args[field] = value

    return args

def configure_kicker_program(SC: SimulatedCommissioning, program_kind: str, program_name: str, program_conf: dict):
    for field in KICK_PROGRAM_MANDATORY_FIELDS: 
        if field not in program_conf:
            raise pySCConfigurationError(f"Mandatory field '{field}' not found in kickers/{program_name} configuration.")
    control = get_kicker_control(SC, program_name=program_name, program_conf=program_conf)
    args = get_kicker_program_arguments(program_conf=program_conf, program_name=program_name, program_kind=program_kind)
    return control, args

def configure_kickers(SC: SimulatedCommissioning) -> None:
    kickers_conf = dict.get(SC.configuration, 'kickers', {})

    for program_kind in kickers_conf.keys():
        if program_kind not in ALLOWED_PROGRAMS:
            raise pySCConfigurationError(f"Unknown kicker program kind: {program_kind}.")

        for program_name in kickers_conf[program_kind]:
            program_conf = kickers_conf[program_kind][program_name]
            control, args = configure_kicker_program(SC=SC,
                                                     program_kind=program_kind,
                                                     program_name=program_name,
                                                     program_conf=program_conf)
            if program_kind == "single_kick":
                SC.kickers.add_single_kick_program(name=program_name, control=control, **args)
                logger.info(f"Single kick program: {program_name} with control: {control}.")
            elif program_kind == "ac":
                SC.kickers.add_ac_program(name=program_name, control=control, **args)
                logger.info(f"ac program: {program_name} with control: {control}.")
            elif program_kind == "white_noise":
                SC.kickers.add_white_noise_program(name=program_name, control=control, **args)
                logger.info(f"White noise program: {program_name} with control: {control}.")
    return
