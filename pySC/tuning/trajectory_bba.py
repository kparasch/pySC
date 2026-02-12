from pydantic import BaseModel
from typing import TYPE_CHECKING, Dict, Literal
import numpy as np
import logging
from ..core.control import IndivControl
from ..apps import measure_bba
from ..apps.bba import BBAAnalysis
from .pySC_interface import pySCInjectionInterface

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..core.simulated_commissioning import SimulatedCommissioning

def get_mag_s_pos(SC: "SimulatedCommissioning", MAG: list[str]):
    s_list = []
    for control_name in MAG:
        control = SC.magnet_settings.controls[control_name]
        if type(control.info) is IndivControl:
            magnet_name = control.info.magnet_name
        else:
            raise NotImplementedError(f"{control} is of type {type(control.info).__name__} which is not implemented.")
        index = SC.magnet_settings.magnets[magnet_name].sim_index
        s_pos = SC.lattice.twiss['s'][index]
        s_list.append(s_pos)
    return s_list

class Trajectory_BBA_Configuration(BaseModel, extra="forbid"):
    config: Dict = dict()

    @classmethod
    def generate_config(cls, SC: "SimulatedCommissioning", max_dx_at_bpm = 1e-3,
                        max_modulation=200e-6, n_shots=1, max_ncorr_index=10,
                        n_downstream_bpms=50):
 #                       max_modulation=600e-6, max_dx_at_bpm=1.5e-3

        config = {}
        n_turns = 1
        RM_name = f'trajectory{n_turns}'
        SC.tuning.fetch_response_matrix(RM_name, orbit=False, n_turns=n_turns)
        RM = SC.tuning.response_matrix[RM_name]

        bba_magnets = SC.tuning.bba_magnets
        bba_magnets_s = get_mag_s_pos(SC, bba_magnets)

        #d1, d2 = RM.RM.shape
        nh = len(SC.tuning.HCORR)
        nbpm = len(SC.bpm_system.indices)
        HRM = RM.matrix[:nbpm, :nh]
        VRM = RM.matrix[nbpm:, nh:]

        for bpm_number in range(len(SC.bpm_system.indices)):
            bpm_index = SC.bpm_system.indices[bpm_number]
            bpm_s = SC.lattice.twiss['s'][bpm_index]

            bba_magnet_number = np.argmin(np.abs(bba_magnets_s - bpm_s))
            the_bba_magnet = bba_magnets[bba_magnet_number]

            HCORR_s = np.array(get_mag_s_pos(SC, SC.tuning.HCORR))
            HCORR_numbers = list(np.where(HCORR_s < bpm_s)[0])
            if len(HCORR_numbers) > max_ncorr_index:
                HCORR_numbers = HCORR_numbers[-max_ncorr_index:]

            VCORR_s = np.array(get_mag_s_pos(SC, SC.tuning.VCORR))
            VCORR_numbers = list(np.where(VCORR_s < bpm_s)[0])
            if len(VCORR_numbers) > max_ncorr_index:
                VCORR_numbers = VCORR_numbers[-max_ncorr_index:]

            max_H_response = -1
            the_HCORR_number = -1
            for nn in HCORR_numbers:
                response = np.abs(HRM[bpm_number, nn])
                if response > max_H_response:
                    max_H_response = float(response)
                    the_HCORR_number = int(nn)
            if max_H_response <= 0:
                logger.warning(f'WARNING: zero H response for BPM {SC.bpm_system.names[bpm_number]}!')
                hcorr_delta = 0
            else:
                hcorr_delta = max_dx_at_bpm/max_H_response

            if the_bba_magnet.split('/')[-1][0] == 'B':
                temp_RM = HRM[bpm_number:bpm_number+n_downstream_bpms, the_HCORR_number]
            else: # it is a skew quadrupole component
                ## TODO: this is wrong if hcorr and vcorr are not the same magnets!!
                temp_RM = VRM[bpm_number:bpm_number+n_downstream_bpms, the_HCORR_number]

            max_response = float(np.max(np.abs(temp_RM)))
            if max_response < 1e-10:
                logger.warning(f'WARNING: very small response for BPM {SC.bpm_system.names[bpm_number]} from magnet {the_bba_magnet} and HCORR {SC.tuning.HCORR[the_HCORR_number]}')
                quad_dk_h = 0
                hcorr_delta = 0
            else:
                quad_dk_h = (max_modulation/max_response) / max_dx_at_bpm

            max_V_response = -1
            the_VCORR_number = -1
            for nn in VCORR_numbers:
                response = np.abs(VRM[bpm_number, nn])
                if response > max_V_response:
                    max_V_response = float(response)
                    the_VCORR_number = int(nn)
            if max_V_response <= 0:
                logger.warning(f'WARNING: zero V response for BPM {SC.bpm_system.names[bpm_number]}!')
                vcorr_delta = 0
            else:
                vcorr_delta = max_dx_at_bpm/max_V_response

            if the_bba_magnet.split('/')[-1][0] == 'B':
                temp_RM = VRM[bpm_number:bpm_number+n_downstream_bpms, the_VCORR_number]
                quad_is_skew = False
            else: # it is a skew quadrupole component
                ## TODO: this is wrong if hcorr and vcorr are not the same magnets!!
                temp_RM = HRM[bpm_number:bpm_number+n_downstream_bpms, the_VCORR_number]
                quad_is_skew = True
            max_response = float(np.max(np.abs(temp_RM)))
            if max_response < 1e-10:
                logger.warning(f'WARNING: very small response for BPM {SC.bpm_system.names[bpm_number]} from magnet {the_bba_magnet} and HCORR {SC.tuning.VCORR[the_VCORR_number]}')
                quad_dk_v = 0
                vcorr_delta = 0
            else:
                quad_dk_v = (max_modulation/max_response) / max_dx_at_bpm

            bpm_name = SC.bpm_system.names[bpm_number]
            config[bpm_name] = {'index': bpm_index,
                                'number': bpm_number,
                                'QUAD': the_bba_magnet,
                                'HCORR_number': the_HCORR_number,
                                'HCORR': SC.tuning.HCORR[the_HCORR_number],
                                'VCORR_number': the_VCORR_number,
                                'VCORR': SC.tuning.VCORR[the_VCORR_number],
                                'HCORR_delta': hcorr_delta,
                                'QUAD_dk_H': quad_dk_h,
                                'VCORR_delta': vcorr_delta,
                                'QUAD_dk_V': quad_dk_v,
                                'QUAD_is_skew': quad_is_skew,
                               }

        return Trajectory_BBA_Configuration(config=config)


def trajectory_bba(SC: "SimulatedCommissioning", bpm_name: str, n_corr_steps: int = 5,
                   n_downstream_bpms: int = 50, plane: Literal['H','V'] = 'H', shots_per_trajectory: int = 1):

    assert plane in ['H', 'V']

    ## get configuration of measurement
    if SC.tuning.trajectory_bba_config is None:
        SC.tuning.generate_trajectory_bba_config()
    config = SC.tuning.trajectory_bba_config.config[bpm_name]

    interface = pySCInjectionInterface(SC=SC, n_turns=2, bba=False, subtract_reference=False)
    generator = measure_bba(interface=interface, bpm_name=bpm_name, config=config,
                            shots_per_orbit=shots_per_trajectory, n_corr_steps=n_corr_steps,
                            bipolar=True, skip_save=True, plane=plane, skip_cycle=True)

    for _, measurement in generator:
        pass

    if plane == 'H':
        data = measurement.H_data
    else:
        data = measurement.V_data

    try:
        analysis_result = BBAAnalysis.analyze(data, n_downstream=n_downstream_bpms)
        offset = analysis_result.offset
        offset_error = analysis_result.offset_error
    except Exception as exc:
        print(exc)
        logger.warning(f'Failed to compute trajectory BBA for BPM {bpm_name}')
        offset, offset_error = 0, np.nan

    return offset, offset_error
