from pydantic import BaseModel
from typing import TYPE_CHECKING, Dict, Literal
import numpy as np
import logging

from ..core.control import IndivControl
from .pySC_interface import pySCOrbitInterface
from ..apps import measure_bba
from ..apps.bba import BBAAnalysis

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
        s_list.append(float(s_pos))
    return s_list

class Orbit_BBA_Configuration(BaseModel, extra="forbid"):
    config: Dict = dict()

    @classmethod
    def generate_config(cls, SC: "SimulatedCommissioning", max_dx_at_bpm = 1e-3,
                        max_modulation=20e-6):

        config = {}
        RM_name = 'orbit'
        SC.tuning.fetch_response_matrix(RM_name, orbit=True)
        RM = SC.tuning.response_matrix[RM_name]

        bba_magnets = SC.tuning.bba_magnets
        bba_magnets_s = get_mag_s_pos(SC, bba_magnets)

        mask_H = np.array(RM.inputs_plane) == 'H'
        mask_V = np.array(RM.inputs_plane) == 'V'
        d1, _ = RM.matrix.shape
        HRM = RM.matrix[:d1//2, mask_H]
        VRM = RM.matrix[d1//2:, mask_V]

        betx = SC.lattice.twiss['betx']
        bety = SC.lattice.twiss['bety']
        qx = SC.lattice.twiss['qx']
        qy = SC.lattice.twiss['qy']
        betx_at_bpms = betx[SC.bpm_system.indices]
        bety_at_bpms = bety[SC.bpm_system.indices]
        for bpm_number in range(len(SC.bpm_system.indices)):
            bpm_index = SC.bpm_system.indices[bpm_number]
            bpm_s = SC.lattice.twiss['s'][bpm_index]

            bba_magnet_number = np.argmin(np.abs(bba_magnets_s - bpm_s))
            the_bba_magnet = bba_magnets[bba_magnet_number]
            bba_magnet_info = SC.magnet_settings.controls[the_bba_magnet].info
            assert type(bba_magnet_info) is IndivControl, f'BBA magnet of unsupported type: {type(bba_magnet_info)}'
            bba_magnet_is_integrated = bba_magnet_info.is_integrated
            bba_magnet_index = SC.magnet_settings.magnets[bba_magnet_info.magnet_name].sim_index
            if bba_magnet_info.component == 'B':
                quad_is_skew = False
            else: # it is a skew quadrupole component
                quad_is_skew = True

            max_H_response = -1
            the_HCORR_number = -1
            response = np.abs(HRM[bpm_number])
            imax = np.argmax(response)
            if response[imax] > max_H_response:
                max_H_response = float(response[imax])
                the_HCORR_number = int(imax)
            hcorr_delta = max_dx_at_bpm/max_H_response

            if not quad_is_skew:
                quad_response = np.sqrt(betx_at_bpms * betx[bba_magnet_index]) / (2 * np.abs(np.sin(np.pi*qx)))
            else: # it is a skew quadrupole component
                quad_response = np.sqrt(bety_at_bpms * bety[bba_magnet_index]) / (2 * np.abs(np.sin(np.pi*qy)))
            quad_dkl_h = (max_modulation / float(np.max(np.abs(quad_response)))) / max_dx_at_bpm

            max_V_response = -1
            the_VCORR_number = -1
            response = np.abs(VRM[bpm_number])
            imax = np.argmax(response)
            if response[imax] > max_V_response:
                max_V_response = float(response[imax])
                the_VCORR_number = int(imax)
            vcorr_delta = max_dx_at_bpm/max_V_response

            if not quad_is_skew:
                quad_response = np.sqrt(bety_at_bpms * bety[bba_magnet_index]) / (2 * np.abs(np.sin(np.pi*qy)))
            else: # it is a skew quadrupole component
                quad_response = np.sqrt(betx_at_bpms * betx[bba_magnet_index]) / (2 * np.abs(np.sin(np.pi*qx)))
            quad_dkl_v = (max_modulation / float(np.max(np.abs(quad_response)))) / max_dx_at_bpm

            if not bba_magnet_is_integrated:
                bba_magnet_length = SC.magnet_settings.magnets[bba_magnet_info.magnet_name].length
                quad_dk_h = quad_dkl_h / bba_magnet_length
                quad_dk_v = quad_dkl_v / bba_magnet_length
            else:
                quad_dk_h = quad_dkl_h
                quad_dk_v = quad_dkl_v

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

        return Orbit_BBA_Configuration(config=config)


def orbit_bba(SC: "SimulatedCommissioning", bpm_name: str, n_corr_steps: int = 7,
                   plane: Literal['H','V'] = 'H', shots_per_orbit: int = 1):

    assert plane in ['H', 'V']

    ## get configuration of measurement
    if SC.tuning.orbit_bba_config is None:
        SC.tuning.generate_orbit_bba_config()
    config = SC.tuning.orbit_bba_config.config[bpm_name]

    interface = pySCOrbitInterface(SC=SC, n_turns=2, bba=False, subtract_reference=False)
    generator = measure_bba(interface=interface, bpm_name=bpm_name, config=config,
                            shots_per_orbit=shots_per_orbit, n_corr_steps=n_corr_steps,
                            bipolar=True, skip_save=True, plane=plane, skip_cycle=True)

    for _, measurement in generator:
        pass

    if plane == 'H':
        data = measurement.H_data
    else:
        data = measurement.V_data

    try:
        analysis_result = BBAAnalysis.analyze(data)
        offset = analysis_result.offset
        offset_error = analysis_result.offset_error
    except Exception as exc:
        print(exc)
        logger.warning(f'Failed to compute trajectory BBA for BPM {bpm_name}')
        offset, offset_error = 0, np.nan

    return offset, offset_error
