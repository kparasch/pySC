from pydantic import BaseModel
from typing import TYPE_CHECKING, Dict, Literal
import numpy as np
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..core.new_simulated_commissioning import SimulatedCommissioning

BPM_OUTLIER = 6 # number of sigma
SLOPE_FACTOR = 0.10 # of max slope
CENTER_OUTLIER = 1 # number of sigma


def get_mag_s_pos(SC: "SimulatedCommissioning", MAG: list[str]):
    s_list = []
    for corr in MAG:
        corr_name = corr.split('/')[0] 
        index = SC.magnet_settings.magnets[corr_name].sim_index
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
        HRM = RM.RM[:nbpm, :nh]
        VRM = RM.RM[nbpm:, nh:]

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
            else: # it is a skew quadrupole component
                ## TODO: this is wrong if hcorr and vcorr are not the same magnets!!
                temp_RM = HRM[bpm_number:bpm_number+n_downstream_bpms, the_VCORR_number]
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
                               }

        return Trajectory_BBA_Configuration(config=config)


def trajectory_bba(SC: "SimulatedCommissioning", bpm_name: str, n_corr_steps: int = 5,
                   n_downstream_bpms: int = 50, plane: Literal['H','V'] = 'H', shots_per_trajectory: int = 1):

    assert plane in ['H', 'V']

    ## get configuration of measurement
    if SC.tuning.trajectory_bba_config is None:
        SC.tuning.generate_trajectory_bba_config()
    config = SC.tuning.trajectory_bba_config.config[bpm_name]

    corr = config[f'{plane}CORR']
    quad = config['QUAD']
    bpm_number = config['number']
    corr_delta_sp = config[f'{plane}CORR_delta']
    quad_delta = config[f'QUAD_dk_{plane}']

    n1 = bpm_number + 1
    n2 = bpm_number + 1 + n_downstream_bpms

    ## define get_orbit
    def get_orbit():
        x, y = SC.bpm_system.capture_injection(n_turns=2, bba=False, subtract_reference=False, use_design=False)
        x = x / shots_per_trajectory
        y = y / shots_per_trajectory
        for i in range(shots_per_trajectory-1):
            x_tmp, y_tmp = SC.bpm_system.capture_injection(n_turns=2, bba=False, subtract_reference=False, use_design=False)
            x = x + x_tmp / shots_per_trajectory
            y = y + y_tmp / shots_per_trajectory

        return (x.flatten(order='F'), y.flatten(order='F'))


    ## define settings to get/set
    settings = SC.magnet_settings

    bpm_pos = np.zeros([n_corr_steps, 2])
    orbits = np.zeros([n_corr_steps, 2, n_downstream_bpms])

    corr_sp0 = settings.get(corr)
    quad_sp0 = settings.get(quad)

    corr_sp_array = np.linspace(-corr_delta_sp, corr_delta_sp, n_corr_steps) + corr_sp0 
    for i_corr, corr_sp in enumerate(corr_sp_array):
        settings.set(corr, corr_sp)
        trajectory_x_center, trajectory_y_center = get_orbit()

        settings.set(quad, quad_sp0 + quad_delta)
        trajectory_x_up, trajectory_y_up = get_orbit()

        settings.set(quad, quad_sp0 - quad_delta)
        trajectory_x_down, trajectory_y_down = get_orbit()

        settings.set(quad, quad_sp0)

        if plane == 'H':
            trajectory_main_down = trajectory_x_down
            trajectory_main_up = trajectory_x_up
            trajectory_main_center = trajectory_x_center
            trajectory_other_down = trajectory_y_down
            trajectory_other_up = trajectory_y_up
            trajectory_other_center = trajectory_y_center
        else:
            trajectory_main_down = trajectory_y_down
            trajectory_main_up = trajectory_y_up
            trajectory_main_center = trajectory_y_center
            trajectory_other_down = trajectory_x_down
            trajectory_other_up = trajectory_x_up
            trajectory_other_center = trajectory_x_center

        bpm_pos[i_corr, 0] = trajectory_main_down[bpm_number]
        bpm_pos[i_corr, 1] = trajectory_main_up[bpm_number]
        if quad.split('/')[-1] == 'B2':
            orbits[i_corr, 0, :] = trajectory_main_down[n1:n2] - trajectory_main_center[n1:n2]
            orbits[i_corr, 1, :] = trajectory_main_up[n1:n2] - trajectory_main_center[n1:n2]
        elif quad.split('/')[-1] == 'A2': ## skew quad
            orbits[i_corr, 0, :] = trajectory_other_down[n1:n2] - trajectory_other_center[n1:n2]
            orbits[i_corr, 1, :] = trajectory_other_up[n1:n2] - trajectory_other_center[n1:n2]
        else:
            raise Exception(f'Invalid magnet for BBA: {quad}')

    settings.set(corr, corr_sp0)
    settings.set(quad, quad_sp0)

    try:
        slopes, slopes_err, center, center_err = get_slopes_center(bpm_pos, orbits, quad_delta)
        mask_bpm_outlier = reject_bpm_outlier(orbits)
        mask_slopes = reject_slopes(slopes)
        mask_center = reject_center_outlier(center)
        final_mask = np.logical_and(np.logical_and(mask_bpm_outlier, mask_slopes), mask_center)

        offset, offset_err = get_offset(center, center_err, final_mask)
    except Exception as exc:
        print(exc)
        logger.warning(f'Failed to compute trajectory BBA for BPM {bpm_name}')
        offset, offset_err = np.nan, np.nan
 
    return offset, offset_err

def reject_bpm_outlier(orbits):
    n_k1 = orbits.shape[1]
    n_bpms = orbits.shape[2]
    mask = np.ones(n_bpms, dtype=bool)
    for k1_step in range(n_k1):
        for bpm in range(n_bpms):
            data = orbits[:, k1_step, bpm]
            if np.any(data - np.mean(data) > BPM_OUTLIER * np.std(data)):
                mask[bpm] = False
 
    # n_rejections = n_bpms - np.sum(mask)
    # print(f"Rejected {n_rejections}/{n_bpms} bpms for bpm outliers ( > {BPM_OUTLIER} r.m.s. )")
    return mask

def reject_slopes(slopes):
    max_slope = np.nanmax(np.abs(slopes))
    mask = np.abs(slopes) > SLOPE_FACTOR * max_slope

    # n_rejections = len(slopes) - np.sum(mask)
    # print(f"Rejected {n_rejections}/{len(slopes)} bpms for small slope ( < {SLOPE_FACTOR} * max(slope) )")
    return mask

def reject_center_outlier(center):
    mean = np.nanmean(center)
    std = np.nanstd(center)
    mask =  abs(center - mean) < CENTER_OUTLIER * std

    # n_rejections = len(center) - np.sum(mask)
    # print(f"Rejected {n_rejections}/{len(center)} bpms for center away from mean ( > {CENTER_OUTLIER} r.m.s. )")
    return mask

def get_slopes_center(bpm_pos, orbits, dk1):
    mag_vec = np.array([dk1, -dk1])
    num_downstream_bpms = orbits.shape[2]
    fit_order = 1
    x = np.mean(bpm_pos, axis=1)
    x_mask = ~np.isnan(x)
    err = np.mean(np.std(bpm_pos[x_mask, :], axis=1))
    x = x[x_mask]
    new_tmp_tra = orbits[x_mask, :, :]

    tmp_slope = np.full((new_tmp_tra.shape[0], new_tmp_tra.shape[2]), np.nan)
    tmp_slope_err = np.full((new_tmp_tra.shape[0], new_tmp_tra.shape[2]), np.nan)
    center = np.full((new_tmp_tra.shape[2]), np.nan)
    center_err = np.full((new_tmp_tra.shape[2]), np.nan)
    for i in range(new_tmp_tra.shape[0]):
        for j in range(new_tmp_tra.shape[2]):
            y = new_tmp_tra[i, :, j]
            y_mask = ~np.isnan(y)
            if np.sum(y_mask) < min(len(mag_vec), 3):
                continue
            # TODO once the position errors are calculated and propagated, should be used
            p, pcov = np.polyfit(mag_vec[y_mask], y[y_mask], 1, w=np.ones(int(np.sum(y_mask))) / err, cov='unscaled')
            tmp_slope[i, j], tmp_slope_err[i, j] = p[0], pcov[0, 0]

    slopes = np.full((new_tmp_tra.shape[2]), np.nan)
    slopes_err = np.full((new_tmp_tra.shape[2]), np.nan)
    for j in range(min(new_tmp_tra.shape[2], num_downstream_bpms)):
        y = tmp_slope[:, j]
        y_err = tmp_slope_err[:, j]
        y_mask = ~np.isnan(y)
        if np.sum(y_mask) <= fit_order + 1:
            continue
        # TODO here do odr as the x values have also measurement errors
        p, pcov = np.polyfit(x[y_mask], y[y_mask], fit_order, w=1 / y_err[y_mask], cov='unscaled')
        if np.abs(p[0]) < 2 * np.sqrt(pcov[0, 0]):
            continue
        center[j] = -p[1] / (fit_order * p[0])  # zero-crossing if linear, minimum is quadratic
        center_err[j] = np.sqrt(center[j] ** 2 * (pcov[0,0]/p[0]**2 + pcov[1,1]/p[1]**2 - 2 * pcov[0, 1] / p[0] / p[1]))
        slopes[j] = p[0]
        slopes_err[j] = np.sqrt(pcov[0,0])

    return slopes, slopes_err, center, center_err

def get_offset(center, center_err, mask):
    from pySC.utils import stats
    try:
        offset_change = stats.weighted_mean(center[mask], center_err[mask])
        offset_change_error = stats.weighted_error(center[mask]-offset_change, center_err[mask]) / np.sqrt(stats.effective_sample_size(center[mask], stats.weights_from_errors(center_err[mask])))
    except ZeroDivisionError as exc:
        print(exc)
        print('Failed to estimate offset!!')
        print(f'Debug info: {center=}, {center_err=}, {mask=}')
        print(f'Debug info: {center[mask]=}, {center_err[mask]=}')
        offset_change = 0
        offset_change_error = np.nan

    return offset_change, offset_change_error
