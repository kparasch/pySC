from pydantic import BaseModel
from typing import TYPE_CHECKING, Dict, Literal, Optional
import numpy as np
import logging
from ..apps import measure_bba
from ..apps.bba import BBAAnalysis
from ..core.control import IndivControl
from ..core.types import MagnetType
from .pySC_interface import pySCInjectionInterface
from .orbit_bba import get_mag_s_pos

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..core.simulated_commissioning import SimulatedCommissioning

class Trajectory_BBA_Configuration(BaseModel, extra="forbid"):
    config: Dict = dict()

    @classmethod
    def generate_config(cls, SC: "SimulatedCommissioning",
                        max_dx_at_bpm: float = 1e-3,
                        max_modulation: float = 200e-6,
                        max_ncorr_index: int = 10,
                        n_downstream_bpms: int = 50,
                        max_dx_at_bpm_sextupole: Optional[float] = None,
                        max_modulation_sextupole: Optional[float] = None,
                        ignore_sextupoles: bool = False):

        if max_modulation_sextupole is None:
            max_modulation_sextupole = max_modulation
        if max_dx_at_bpm_sextupole is None:
            max_dx_at_bpm_sextupole = max_dx_at_bpm

        config = {}
        n_turns = 1
        RM_name = f'trajectory{n_turns}'
        SC.tuning.fetch_response_matrix(RM_name, orbit=False, n_turns=n_turns)
        RM = SC.tuning.response_matrix[RM_name]

        all_bba_magnets = SC.tuning.bba_magnets
        if ignore_sextupoles:
            bba_magnets = []
            for control in all_bba_magnets:
                info = SC.magnet_settings.controls[control].info 
                if not ( info.component == "B" and info.order == 3 ):
                    bba_magnets.append(control)
        else:
            bba_magnets = all_bba_magnets

        bba_magnets_s = get_mag_s_pos(SC, bba_magnets)

        #d1, d2 = RM.RM.shape
        nh = len(SC.tuning.HCORR)
        nbpm = len(SC.bpm_system.indices)
        HRM = RM.matrix[:nbpm, :nh]
        VRM = RM.matrix[nbpm:, nh:]

        betx = SC.lattice.twiss['betx']
        bety = SC.lattice.twiss['bety']
        mux = SC.lattice.twiss['mux']
        muy = SC.lattice.twiss['muy']

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
                if bba_magnet_info.order == 2:
                    magnet_type = MagnetType.norm_quad
                elif bba_magnet_info.order == 3:
                    magnet_type = MagnetType.norm_sext
                else:
                    raise NotImplementedError("BBA configuration for {bba_magnet_info.component}{bba_magnet_info.order} magnets not implemented.")
            else: # it is a skew quadrupole component
                if bba_magnet_info.order == 2:
                    magnet_type = MagnetType.skew_quad
                else:
                    raise NotImplementedError("BBA configuration for {bba_magnet_info.component}{bba_magnet_info.order} magnets not implemented.")

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
                if magnet_type in [MagnetType.norm_quad, MagnetType.skew_quad]:
                    hcorr_delta = max_dx_at_bpm/max_H_response
                elif magnet_type is MagnetType.norm_sext:
                    hcorr_delta = max_dx_at_bpm_sextupole/max_H_response

            downstream_bpm_indices = SC.bpm_system.indices[bpm_number:bpm_number+n_downstream_bpms]

            target_betx = betx[downstream_bpm_indices]
            target_bety = bety[downstream_bpm_indices]
            target_mux = mux[downstream_bpm_indices]
            target_muy = muy[downstream_bpm_indices]

            source_betx = betx[bba_magnet_index]
            source_bety = bety[bba_magnet_index]
            source_mux = mux[bba_magnet_index]
            source_muy = muy[bba_magnet_index]

            # make response zero in bpms upstream of bba magnet by setting equal phase advances
            target_mux[target_mux < source_mux] = source_mux 
            target_muy[target_muy < source_muy] = source_muy 

            # sign depends on several things but we take absolute value later, so we ignore it here
            bba_magnet_response_x = np.sqrt(target_betx * source_betx) * np.sin(2*np.pi*(target_mux - source_mux))
            bba_magnet_response_y = np.sqrt(target_bety * source_bety) * np.sin(2*np.pi*(target_muy - source_muy))

            if magnet_type is MagnetType.norm_quad: 
                max_response = float(np.max(np.abs(bba_magnet_response_x)))
            elif magnet_type is MagnetType.skew_quad:
                max_response = float(np.max(np.abs(bba_magnet_response_y)))
            elif magnet_type is MagnetType.norm_sext:
                max_response = float(np.max(np.abs(bba_magnet_response_x)))

            if max_response < 1e-10:
                logger.warning(f'WARNING: very small response for BPM {SC.bpm_system.names[bpm_number]} from magnet {the_bba_magnet} and HCORR {SC.tuning.HCORR[the_HCORR_number]}')
                quad_dkl_h = 0
                hcorr_delta = 0
            else:
                if magnet_type in [MagnetType.norm_quad, MagnetType.skew_quad]:
                    quad_dkl_h = (max_modulation/max_response) / max_dx_at_bpm
                elif magnet_type is MagnetType.norm_sext:
                    quad_dkl_h = 2*(max_modulation_sextupole/max_response) / max_dx_at_bpm_sextupole**2



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
                if magnet_type in [MagnetType.norm_quad, MagnetType.skew_quad]:
                    vcorr_delta = max_dx_at_bpm/max_V_response
                elif magnet_type is MagnetType.norm_sext:
                    vcorr_delta = max_dx_at_bpm_sextupole/max_V_response

            if magnet_type is MagnetType.norm_quad: 
                max_response = float(np.max(np.abs(bba_magnet_response_y)))
            elif magnet_type is MagnetType.skew_quad:
                max_response = float(np.max(np.abs(bba_magnet_response_x)))
            elif magnet_type is MagnetType.norm_sext:
                max_response = float(np.max(np.abs(bba_magnet_response_x)))

            if max_response < 1e-10:
                logger.warning(f'WARNING: very small response for BPM {SC.bpm_system.names[bpm_number]} from magnet {the_bba_magnet} and HCORR {SC.tuning.VCORR[the_VCORR_number]}')
                quad_dkl_v = 0
                vcorr_delta = 0
            else:
                if magnet_type in [MagnetType.norm_quad, MagnetType.skew_quad]:
                    quad_dkl_v = (max_modulation/max_response) / max_dx_at_bpm
                elif magnet_type is MagnetType.norm_sext:
                    quad_dkl_v = 2*(max_modulation_sextupole/max_response) / max_dx_at_bpm_sextupole**2

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
                                'magnet_type': magnet_type.value,
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
        logger.warning(f"Exception | {type(exc).__name__}: {exc}")
        logger.warning(f'Failed to compute trajectory BBA for BPM {bpm_name}')
        offset, offset_error = 0, np.nan

    return offset, offset_error
