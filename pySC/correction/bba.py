import matplotlib.pyplot as plt
import numpy as np

from pySC.core.beam import bpm_reading, all_elements_reading
from pySC.core.classes import DotDict
from pySC.core.constants import TRACK_TBT, NUM_TO_AB, SETTING_REL, SETTING_ABS
from pySC.utils import at_wrapper, logging_tools, sc_tools, stats
from pySC.correction import orbit_trajectory
from pySC.lattice_properties.response_model import SCgetModelRM


LOGGER = logging_tools.get_logger(__name__)
CONFIDENCE_LEVEL_SIGMAS = 2.5
OUTLIER_LIMIT = 1.5e-3
# TRAJECTORY_BBA_THRESHOLD_LEVEL = 700e-6
# TRAJECTORY_BBA_THRESHOLD_GAIN = 0.5


def trajectory_bba(SC, bpm_ords, mag_ords, print_true_offset=True, **kwargs):
    par = DotDict(dict(n_steps=10, fit_order=1, magnet_order=1, skewness=False, setpoint_method=SETTING_REL,
                       q_ord_phase=np.array([], dtype=int), q_ord_setpoints=np.ones(1), magnet_strengths=np.array([0.95, 1.05]),
                       num_downstream_bpms=len(SC.ORD.BPM), max_injection_pos_angle=np.array([0.9E-3, 0.9E-3]),
                       dipole_compensation=True, plot_results=False))
    par.update(**kwargs)
    par = _check_input(bpm_ords, mag_ords, par)
    if SC.INJ.trackMode != TRACK_TBT or SC.INJ.nTurns != 2:
        raise ValueError('Beam-trajectory-based alignment works in TBT mode with 2 turns. '
                         'Please set: SC.INJ.nTurns = 2 and SC.INJ.trackMode = "TBT"')

    q0 = at_wrapper.atgetfieldvalues(SC.RING, par.q_ord_phase, "SetPointB", index=1)
    bba_offsets = np.full(bpm_ords.shape, np.nan)
    bba_offset_errors = np.full(bpm_ords.shape, np.nan)

    if print_true_offset:
        true_offsets = np.full(bpm_ords.shape, np.nan)

    for n_dim in range(bpm_ords.shape[0]):  # TODO currently assumes either horizontal or both planes
        last_bpm_ind = np.where(bpm_ords[n_dim, -1] == SC.ORD.BPM)[0][0]
        quads_strengths, scalings, bpm_ranges = _phase_advance_injection_scan(SC, n_dim, last_bpm_ind, par)
        LOGGER.info(f'Scanned plane {n_dim}')
        for j_bpm in range(bpm_ords.shape[1]):  # j_bpm: Index of BPM adjacent to magnet for BBA
            LOGGER.info(f'BPM number {j_bpm}')
            bpm_index = np.where(bpm_ords[n_dim, j_bpm] == SC.ORD.BPM)[0][0]
            m_ord = mag_ords[n_dim, j_bpm]
            set_ind = np.argmax(bpm_ranges[:, bpm_index])
            SC.set_magnet_setpoints(par.q_ord_phase, quads_strengths[set_ind], False, 1, method=SETTING_REL, dipole_compensation=True)
            bpm_pos, downstream_trajectories = _data_measurement_tbt(SC, m_ord, bpm_index, j_bpm, n_dim, scalings[set_ind], par)
            if len(q0):
                SC.set_magnet_setpoints(par.q_ord_phase, q0, False, 1, method=SETTING_ABS, dipole_compensation=True)
                try:
                    bba_offsets[n_dim, j_bpm], bba_offset_errors[n_dim, j_bpm] = _data_evaluation(SC, bpm_pos, downstream_trajectories, par.magnet_strengths[n_dim, j_bpm], n_dim, m_ord, par)
                except Exception as e:
                    LOGGER.info(f'BPM number {j_bpm} threw exception! {e}')
                    bba_offsets[n_dim, j_bpm] = np.nan
                    bba_offset_errors[n_dim, j_bpm] = np.nan
                # if abs(bba_offsets[n_dim, j_bpm]) > TRAJECTORY_BBA_THRESHOLD_LEVEL:
                #     bba_offsets[n_dim, j_bpm] *= TRAJECTORY_BBA_THRESHOLD_GAIN
                #     LOGGER.info(f'Threshold exceeded, reducing offset to {bba_offsets[n_dim, j_bpm]*1e6:.1f}')
                if is_bba_errored(bba_offsets[n_dim, j_bpm], bba_offset_errors[n_dim, j_bpm]): 
                    LOGGER.info(f'BPM number {j_bpm} failed!')

                bba_offset = bba_offsets[n_dim, j_bpm]
                bpm_id = bpm_ords[n_dim, j_bpm]
                true_offset = ( (SC.RING[m_ord].MagnetOffset[n_dim] + SC.RING[m_ord].SupportOffset[n_dim] ) 
                               -(SC.RING[bpm_id].Offset[n_dim]      + SC.RING[bpm_id].SupportOffset[n_dim]) )  
                true_offsets[n_dim, j_bpm] = true_offset
                if print_true_offset:
                    LOGGER.info(f'BPM number {j_bpm} (id: {bpm_index}), plane {n_dim}, Offsets: Estimated: {(bba_offset)*1e6:.1f} μm, '
                                f'True: {true_offset*1e6:.1f} μm, Δ: {(bba_offset-true_offset)*1e6:.1f} μm')
                else:
                    LOGGER.info(f'BPM number {j_bpm} (id: {bpm_index}), plane {n_dim}, Estimated offset: {(bba_offset)*1e6:.1f} μm')
        if print_true_offset:
            failed = is_bba_errored(bba_offsets[n_dim], bba_offset_errors[n_dim])
            bad_predictions = np.sum(np.abs(true_offsets[n_dim, :][~failed] - bba_offsets[n_dim, :][~failed]) > np.abs(true_offsets[n_dim, :][~failed]))
            LOGGER.info(f'R.m.s. offsets before BBA, dim {n_dim}: {np.nanstd(true_offsets[n_dim, :][~failed]) * 1e6:.1f} μm')
            LOGGER.info(f'R.m.s. offsets after BBA, dim {n_dim}: {np.nanstd(true_offsets[n_dim, :][~failed] - bba_offsets[n_dim, :][~failed]) * 1e6:.1f} μm')
            LOGGER.info(f'Abs. max. error before BBA, dim {n_dim}: {np.max(np.abs(true_offsets[n_dim, :][~failed])) * 1e6:.1f} μm')
            LOGGER.info(f'Abs. max. error after BBA, dim {n_dim}: {np.max(np.abs(true_offsets[n_dim, :][~failed] - bba_offsets[n_dim, :][~failed])) * 1e6:.1f} μm')
            LOGGER.info(f'Number of BPMs with worse offset after BBA: {bad_predictions}')
    SC = apply_bpm_offsets(SC, bpm_ords, bba_offsets, bba_offset_errors)
    if par.plot_results:
        plot_bba_results(SC, bpm_ords, bba_offsets, bba_offset_errors)
    return SC, bba_offsets, bba_offset_errors

def orbit_bba_one_bpm_one_plane(SC, bpm_id, magnet_id, n_dim, n_k1_steps=5, max_dk1=10e-6, max_x=None,
                                n_k2_steps=2, max_dk2=5e-3, quad_is_skew=False, RM=None, print_true_offset=True,
                                use_bump=False):
    
    if RM is None:
        raise NotImplementedError('RM is not provided.')

    if quad_is_skew:
        meas_dim = 1 - n_dim #look for orbit modulation in the other plane when cycling a skew quadrupole
    else:
        meas_dim = n_dim

    bpm_index = np.where(bpm_id == SC.ORD.BPM)[0][0]

    RMi = bpm_index + n_dim * len(SC.ORD.BPM)
    if n_dim == 0:
        reduced_RM = RM[RMi, :len(SC.ORD.CM[0])]
    else:
        reduced_RM = RM[RMi, len(SC.ORD.CM[0]):]

    # select corrector based on maximum effect from response matrix
    corrector_RMindex = np.argmax(np.abs(reduced_RM))
    corrector_index = SC.ORD.CM[n_dim][corrector_RMindex]

    if max_x is not None:
        assert max_dk1 is None
        max_dk1 = max_x/reduced_RM[corrector_RMindex]
    else:
        assert max_dk1 is not None

    if not use_bump:
        corrector_ids = np.array([corrector_index])


    LOGGER.info(f'max kick: {max_dk1*1e6:.2f} μrad, expected max excursion: {max_dk1*reduced_RM[corrector_RMindex]*1e6:.1f} μm')
    cm_is_skew = False if n_dim == 0 else True

    # Get initial state
    zero_kick = SC.get_cm_setpoints(corrector_ids, skewness=cm_is_skew)
    if quad_is_skew:
        zero_quad = SC.RING[magnet_id].SetPointA[1]
    else:
        zero_quad = SC.RING[magnet_id].SetPointB[1]

    k1_steps = np.linspace(-max_dk1, max_dk1, n_k1_steps) + zero_kick
    k2_steps = np.linspace(-max_dk2, max_dk2, n_k2_steps) + zero_quad

    orbits = np.full((n_k1_steps, n_k2_steps, len(SC.ORD.BPM)), np.nan)
    bpm_pos = np.full((n_k1_steps, n_k2_steps), np.nan)

    for k1_step in range(n_k1_steps):
        SC.set_cm_setpoints(corrector_ids, k1_steps[k1_step], skewness=cm_is_skew, method='abs')
        SC.set_magnet_setpoints(magnet_id, zero_quad, skewness=quad_is_skew, order=1, method='abs')
        reference_bpm_reading = bpm_reading(SC)[0]
        for k2_step in range(n_k2_steps):
            SC.set_magnet_setpoints(magnet_id, k2_steps[k2_step], skewness=quad_is_skew, order=1, method='abs') 
            bpm_readings = bpm_reading(SC)[0]
            bpm_pos[k1_step, k2_step] = bpm_readings[n_dim, bpm_index]
            orbits[k1_step, k2_step, :] = bpm_readings[meas_dim, :] - reference_bpm_reading[meas_dim, :]
    LOGGER.info(f'Actual excursions: [{np.min(bpm_pos)*1e6:.1f}, {np.max(bpm_pos)*1e6:.1f}] μm, number of NaNs: {np.sum(np.isnan(bpm_pos))}')

    # revert to initial state
    SC.set_cm_setpoints(corrector_ids, zero_kick, skewness=cm_is_skew, method='abs')
    SC.set_magnet_setpoints(magnet_id, zero_quad, skewness=quad_is_skew, order=1, method='abs')

    # SC.orbits.append(orbits)
    # SC.bps.append(bpm_pos)

    par = DotDict(dict(fit_order=1,# magnet_strengths=k2_steps, 
                       dipole_compensation=True, 
                       num_downstream_bpms=len(SC.ORD.BPM)))
    bba_offset, bba_offset_error = _data_evaluation(SC, bpm_pos, orbits, k2_steps, None, None, par)
    true_offset = ( (SC.RING[magnet_id].MagnetOffset[n_dim] + SC.RING[magnet_id].SupportOffset[n_dim] ) 
                   -(SC.RING[bpm_id].Offset[n_dim]      + SC.RING[bpm_id].SupportOffset[n_dim]) )  

    if print_true_offset:
        LOGGER.info(f'BPM id: {bpm_index}, plane {n_dim}, Offsets: Estimated: {(bba_offset)*1e6:.1f} μm, '
                    f'True: {true_offset*1e6:.1f} μm, Δ: {(bba_offset-true_offset)*1e6:.1f} μm')
    else:
        LOGGER.info(f'BPM id: {bpm_index}, plane {n_dim}, Estimated offset: {(bba_offset)*1e6:.1f} μm')

    return bba_offset, bba_offset_error

def orbit_bba(SC, bpm_ords, mag_ords, quad_is_skew=False, n_k1_steps=5, max_dk1=10e-6, n_k2_steps=2, max_dk2=5e-3,
              RM=None, max_x=None, plot_results=False):
    if bpm_ords.shape != mag_ords.shape:  # both in shape 2 x N
        raise ValueError('Input arrays for BPMs and magnets must be same size.')

    if SC.INJ.trackMode == TRACK_TBT:
        raise ValueError('Beam-orbit-based alignment does not work in TBT mode. '
                         'Please set: SC.INJ.trackMode to  "ORB" or "PORB".')
    bba_offsets = np.full(bpm_ords.shape, np.nan)
    bba_offset_errors = np.full(bpm_ords.shape, np.nan)

    if max_x is not None and max_dk1 is not None:
        raise Exception('Only one of max_x and max_dk1 can be provided.')

    if RM is None:
        LOGGER.info('Response matrix not given, calculating it now.')
        RM = SCgetModelRM(SC, SC.ORD.BPM, SC.ORD.CM, trackMode='ORB', useIdealRing=True)
    n_bpms = bpm_ords.shape[1]

    for j_bpm in range(n_bpms):
        for n_dim in [0, 1]:
            bpm_id = bpm_ords[n_dim, j_bpm]
            magnet_id = mag_ords[n_dim, j_bpm]
            try:
                bba_offsets[n_dim, j_bpm], bba_offset_errors[n_dim, j_bpm] = orbit_bba_one_bpm_one_plane(SC, 
                                bpm_id, magnet_id, n_dim, n_k1_steps=n_k1_steps, max_dk1=max_dk1, max_x=max_x,
                                n_k2_steps=n_k2_steps, max_dk2=max_dk2, quad_is_skew=quad_is_skew, RM=RM)
            except Exception as e:
                LOGGER.info(f'BPM number {j_bpm} failed! {e}')
                bba_offsets[n_dim, j_bpm] = np.nan
                bba_offset_errors[n_dim, j_bpm] = np.nan

    SC = apply_bpm_offsets(SC, bpm_ords, bba_offsets, bba_offset_errors)
    if plot_results:
        plot_bba_results(SC, bpm_ords, bba_offsets, bba_offset_errors)
    return SC, bba_offsets, bba_offset_errors


def fake_bba(SC, bpm_ords, mag_ords, errors=None, fake_offset=None):
    """Example use:
    SC = fake_bba(SC, bpm_ords, mag_ords, is_bba_errored(bba_offsets, bba_offset_errors))"""
    if errors is None and fake_offset is None:
        raise ValueError("At least one of error_flags or fake_offset has to be given")
    if errors is None:
        errors = np.ones(bpm_ords.shape, dtype=bool)
    if fake_offset is None:
        final_offset_errors = _get_bpm_offset_from_mag(SC.RING, bpm_ords, mag_ords)
        final_offset_errors[errors != 0] = np.nan
        fake_offset = np.sqrt(np.nanmean(final_offset_errors ** 2, axis=1))

    LOGGER.info(f"Final offset error is {1E6 * fake_offset} um (hor|ver)"
                f" with {np.sum(errors, axis=1)} measurement failures -> being re-calculated now.\n")
    for inds in np.argwhere(errors):  # TODO get to the form such that apply_bpm_offsets can be used
        fake_bpm_offset = (SC.RING[mag_ords[inds[0], inds[1]]].MagnetOffset[inds[0]]
                           + SC.RING[mag_ords[inds[0], inds[1]]].SupportOffset[inds[0]]
                           - SC.RING[bpm_ords[inds[0], inds[1]]].SupportOffset[inds[0]]
                           + fake_offset[inds[0]] * sc_tools.randnc(2, ()))
        if not np.isnan(fake_bpm_offset):
            SC.RING[bpm_ords[inds[0], inds[1]]].Offset[inds[0]] = fake_bpm_offset
        else:
            LOGGER.info('BPM offset not reassigned, NaN.\n')
    return SC


def _check_input(bpm_ords, mag_ords, par):
    if bpm_ords.shape != mag_ords.shape:  # both in shape 2 x N
        raise ValueError('Input arrays for BPMs and magnets must be same size.')
    if par.magnet_strengths.ndim < 2:
        par.magnet_strengths = np.tile(par.magnet_strengths, mag_ords.shape + (1,))
    if par.fit_order > 2:
        raise ValueError("At most second order fit is supported.")
    return par


def apply_bpm_offsets(SC, bpm_ords, bba_offsets, bba_offset_errors):
    errors = is_bba_errored(bba_offsets, bba_offset_errors)
    # bpm_ords, bba_offsets, bba_offset_errors should have same shape
    for inds in np.argwhere(~errors):
        SC.RING[bpm_ords[inds[0], inds[1]]].Offset[inds[0]] += bba_offsets[inds[0], inds[1]]
    for inds in np.argwhere(errors):
        LOGGER.info(f"Poor resolution for BPM {inds[1]} in plane {inds[0]}: "
                    f"{bba_offsets[inds[0], inds[1]]}+-{bba_offset_errors[inds[0], inds[1]]}")
    return SC
     

def is_bba_errored(bba_offsets, bba_offset_errors):
    return np.logical_or(np.logical_or(np.isnan(bba_offsets), np.abs(bba_offsets) > OUTLIER_LIMIT),
                         np.abs(bba_offsets) < CONFIDENCE_LEVEL_SIGMAS * bba_offset_errors)


def _get_bpm_offset_from_mag(ring, bpm_ords, mag_ords):
    offset = np.full(bpm_ords.shape, np.nan)
    for n_dim in range(bpm_ords.shape[0]):
        offset[n_dim, :] = (at_wrapper.atgetfieldvalues(ring, bpm_ords[n_dim, :], 'Offset', n_dim)
                            + at_wrapper.atgetfieldvalues(ring, bpm_ords[n_dim, :], 'SupportOffset', n_dim)
                            - at_wrapper.atgetfieldvalues(ring, mag_ords[n_dim, :], 'MagnetOffset', n_dim)
                            - at_wrapper.atgetfieldvalues(ring, mag_ords[n_dim, :], 'SupportOffset', n_dim))
    return offset


def _data_evaluation(SC, bpm_pos, trajectories, mag_vec, n_dim, m_ord, par):
    x = np.mean(bpm_pos, axis=1)
    x_mask = ~np.isnan(x)
    err = np.mean(np.std(bpm_pos[x_mask, :], axis=1))
    x = x[x_mask]
    new_tmp_tra = trajectories[x_mask, :, :]

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
    for j in range(min(new_tmp_tra.shape[2], par.num_downstream_bpms)):
        y = tmp_slope[:, j]
        y_err = tmp_slope_err[:, j]
        y_mask = ~np.isnan(y)
        if np.sum(y_mask) <= par.fit_order + 1:
            continue
        # TODO here do odr as the x values have also measurement errors
        p, pcov = np.polyfit(x[y_mask], y[y_mask], par.fit_order, w=1 / y_err[y_mask], cov='unscaled')
        if np.abs(p[0]) < 2 * np.sqrt(pcov[0, 0]):
            continue
        center[j] = -p[1] / (par.fit_order * p[0])  # zero-crossing if linear, minimum is quadratic
        center_err[j] = np.sqrt(center[j] ** 2 * (pcov[0,0]/p[0]**2 + pcov[1,1]/p[1]**2 - 2 * pcov[0, 1] / p[0] / p[1]))
        slopes[j] = p[0]
    mask = ~np.isnan(center)
    nn_b = np.sum(mask)
    max_slope = sorted(abs(slopes))[-3]
    mask = np.logical_and(mask, np.abs(slopes) > 0.1 * max_slope)
    nn_a = np.sum(mask)
    LOGGER.info(f'rejected {nn_b - nn_a} BPMs due to slope')
    offset_change = stats.weighted_mean(center[mask], center_err[mask])
    offset_change_error = stats.weighted_error(center[mask]-offset_change, center_err[mask]) / np.sqrt(stats.effective_sample_size(center[mask], stats.weights_from_errors(center_err[mask])))
    if not par.dipole_compensation and n_dim == 0 and SC.RING[m_ord].NomPolynomB[1] != 0:
        offset_change += getattr(SC.RING[m_ord], 'BendingAngle', 0) / SC.RING[m_ord].NomPolynomB[1] / SC.RING[m_ord].Length
    return offset_change, offset_change_error


def plot_bba_results(SC, bpm_ords, bba_offsets, bba_offset_errors):
    to_um = 1E6
    errors = is_bba_errored(bba_offsets, bba_offset_errors)
    plt.rcParams.update({'font.size': 10})
    f, ax = plt.subplots(nrows=2, num=90, figsize=(8, 8), facecolor="w")
    plabels = ("Horizontal", "Vertical")
    for n_dim in range(bpm_ords.shape[0]):
        x = np.where(np.in1d(SC.ORD.BPM, bpm_ords[n_dim, :]))[0]
        mask = errors[n_dim, :]
        ax[n_dim].errorbar(x[~mask], to_um * bba_offsets[n_dim, ~mask], yerr=to_um * bba_offset_errors[n_dim, ~mask], fmt="bo",
                           label=plabels[n_dim])
        ax[n_dim].errorbar(x[mask], to_um * bba_offsets[n_dim, mask], yerr=to_um * bba_offset_errors[n_dim, mask], fmt="rX",
                           label=f"{plabels[n_dim]} failed")
        ax[n_dim].set_ylabel(r' Offset $\Delta$ [$\mu$m]')
        ax[n_dim].set_xlabel('Index of BPM')
        ax[n_dim].set_xlim([0, len(SC.ORD.BPM)])
        ax[n_dim].legend()
    f.show()


def plot_bpm_offsets_from_magnets(SC, bpm_ords, mag_ords, error_flags):
    plt.rcParams.update({'font.size': 10})
    fom = _get_bpm_offset_from_mag(SC.RING, bpm_ords, mag_ords)
    n_steps = 1 if bpm_ords.shape[1] == 1 else 1.1 * np.max(np.abs(fom)) * np.linspace(-1, 1, int(np.floor(bpm_ords.shape[1] / 3)))
    f, ax = plt.subplots(nrows=3, num=91, figsize=(8, 11), facecolor="w")
    colors = ['#1f77b4', '#ff7f0e']
    for n_dim in range(bpm_ords.shape[0]):
        a, b = np.histogram(fom[n_dim, :], n_steps)
        ax[0].plot(1E6 * b[1:], a, linewidth=2)

    if bpm_ords.shape[0] > 1:
        rmss = 1E6 * np.sqrt(np.nanmean(np.square(fom), axis=1))
        legends = [f"Horizontal rms: {rmss[0]:.0f}$\\mu m$",
                   f"Vertical rms:  {rmss[1]:.0f}$\\mu m$"]
        ax[0].legend(legends)
    ax[0].set_xlabel(r'BPM offset w.r.t. magnet [$\mu$m]')
    ax[0].set_ylabel('Occurrences')

    plabels = ("Horizontal", "Vertical")
    for n_dim in range(bpm_ords.shape[0]):
        x = np.where(np.in1d(SC.ORD.BPM, bpm_ords[n_dim, :]))[0]
        mask = ~error_flags[n_dim, :]
        if not np.all(np.isnan(fom[n_dim, mask])):
            ax[n_dim + 1].plot(x[mask], 1E6 * fom[n_dim, mask], 'bo', label=plabels[n_dim])
        if not np.all(np.isnan(fom[n_dim, ~mask])):
            ax[n_dim + 1].plot(x[~mask], 1E6 * fom[n_dim, ~mask], 'rX', label=f"{plabels[n_dim]} failed")

        ax[n_dim + 1].set_ylabel(r'Offset [$\mu$m]')
        ax[n_dim + 1].set_xlabel('Index of BPM')
        ax[n_dim + 1].set_xlim((0, len(SC.ORD.BPM)))
        ax[n_dim + 1].legend()
    f.tight_layout()
    f.show()


    
# trajectory BBA helper functions


def _data_measurement_tbt(SC, m_ord, bpm_ind, j_bpm, n_dim, scaling, par):
    kick_vec = scaling * par.max_injection_pos_angle.reshape(2, 1) * np.linspace(-1, 1, par.n_steps)
    meas_dim = 1 - n_dim if par.skewness else n_dim
    initial_z0 = SC.INJ.Z0.copy()
    trajectories = np.full((par.n_steps, len(par.magnet_strengths[n_dim, j_bpm]), par.num_downstream_bpms), np.nan)
    bpm_pos = np.full((par.n_steps, len(par.magnet_strengths[n_dim, j_bpm])), np.nan)
    init_setpoint = getattr(SC.RING[m_ord], f"SetPoint{NUM_TO_AB[int(par.skewness)]}")[par.magnet_order]
    for n_q, setpoint_q in enumerate(par.magnet_strengths[n_dim, j_bpm]):
        SC.set_magnet_setpoints(m_ord, setpoint_q, par.skewness, par.magnet_order,
                                method=par.setpoint_method, dipole_compensation=par.dipole_compensation)
        for step in range(par.n_steps):
            SC.INJ.Z0[2 * n_dim:2 * n_dim + 2] = initial_z0[2 * n_dim:2 * n_dim + 2] + kick_vec[:, step]
            bpm_readings = bpm_reading(SC)[0]
            bpm_pos[step, n_q] = bpm_readings[n_dim, bpm_ind]
            trajectories[step, n_q, :] = bpm_readings[meas_dim, bpm_ind:(bpm_ind + par.num_downstream_bpms)]
        if par.setpoint_method == SETTING_ABS:
            SC.set_magnet_setpoints(m_ord, init_setpoint, par.skewness, par.magnet_order,
                                method=SETTING_ABS, dipole_compensation=par.dipole_compensation)

    SC.INJ.Z0 = initial_z0
    SC.set_magnet_setpoints(m_ord, init_setpoint, par.skewness, par.magnet_order,
                            method=SETTING_ABS, dipole_compensation=par.dipole_compensation)
    return bpm_pos, trajectories


def _phase_advance_injection_scan(SC, n_dim, last_bpm_ind, par):
    q0 = at_wrapper.atgetfieldvalues(SC.RING, par.q_ord_phase, "SetPointB", index=1)
    n_setpoints = len(par.q_ord_setpoints)
    bpm_ranges = np.zeros((n_setpoints, len(SC.ORD.BPM)))
    scalings = np.zeros(n_setpoints)
    for i, q_scale in enumerate(par.q_ord_setpoints):
        SC.set_magnet_setpoints(par.q_ord_phase, q_scale, False, 1, method=SETTING_REL, dipole_compensation=True)
        scalings[i], bpm_ranges[i] = _scale_injection_to_reach_bpms(SC, n_dim, last_bpm_ind, par.max_injection_pos_angle)
        if len(q0):
            SC.set_magnet_setpoints(par.q_ord_phase, q0, False, 1, method=SETTING_ABS, dipole_compensation=True)
    return par.q_ord_setpoints, scalings, bpm_ranges


def _scale_injection_to_reach_bpms(SC, n_dim, last_bpm_ind, max_injection_pos_angle):
    initial_z0, initial_nturns = SC.INJ.Z0.copy(), SC.INJ.nTurns
    SC.INJ.nTurns = 1
    scaling_factor = 1.0
    mask = np.ones(len(SC.ORD.BPM), dtype=bool)
    if last_bpm_ind + 1 < len(SC.ORD.BPM):
        mask[last_bpm_ind + 1:] = False
    
    # trying the largest kicks with both signs, if fails, scale down and try again
    for _ in range(20):
        LOGGER.info(f"{scaling_factor=}")
        tmp_bpm_pos = np.full((2, len(SC.ORD.BPM)), np.nan)
        for sign_ind in range(2):
            SC.INJ.Z0[2 * n_dim:2 * n_dim + 2] = initial_z0[2 * n_dim:2 * n_dim + 2] + (-1) ** sign_ind * scaling_factor * max_injection_pos_angle
            tmp_bpm_pos[sign_ind, :] = bpm_reading(SC)[0][n_dim, :]
        SC.INJ.Z0 = initial_z0.copy()

        if np.sum(np.isnan(tmp_bpm_pos[:, mask])) == 0:
            bpm_ranges = np.max(tmp_bpm_pos, axis=0) - np.min(tmp_bpm_pos, axis=0)
            LOGGER.debug(f'Initial trajectory variation scaled to [{100 * scaling_factor}| '
                         f'{100 * scaling_factor}]% of its initial value, '
                         f'BBA-BPM range from {1E6 * np.min(bpm_ranges):.0f} um to {1E6 * np.max(bpm_ranges):.0f}.')
            SC.INJ.nTurns = initial_nturns
            return scaling_factor, bpm_ranges
        scaling_factor *= 0.8
    else:
        LOGGER.warning('Something went wrong. No beam transmission at all(?)')
        SC.INJ.nTurns = initial_nturns
        return scaling_factor, np.zeros(len(SC.ORD.BPM))

# orbit BBA helper functions


# def _data_measurement_orb(SC, m_ord, bpm_ind, j_bpm, n_dim, par, cm_ords, cm_vec):
#     meas_dim = 1 - n_dim if par.skewness else n_dim
#     initial_z0 = SC.INJ.Z0.copy()
#     n_msteps = cm_vec[n_dim].shape[0]
#     orbits = np.full((n_msteps, len(par.magnet_strengths[n_dim, j_bpm]), len(SC.ORD.BPM)), np.nan)
#     bpm_pos = np.full((n_msteps, len(par.magnet_strengths[n_dim, j_bpm])), np.nan)
#     for n_q, setpoint_q in enumerate(par.magnet_strengths[n_dim, j_bpm]):
#         SC.set_magnet_setpoints(m_ord, setpoint_q, par.skewness, par.magnet_order, method=par.setpoint_method,
#                                 dipole_compensation=par.dipole_compensation)
#         for step in range(n_msteps):
#             for n_d in range(2):
#                 SC.set_cm_setpoints(cm_ords[n_d], cm_vec[n_d][step, :], bool(n_d), method=SETTING_ABS)
#             bpm_readings = bpm_reading(SC)[0]
#             bpm_pos[step, n_q] = bpm_readings[n_dim, bpm_ind]
#             orbits[step, n_q, :] = bpm_readings[meas_dim, :]
# 
#     SC.INJ.Z0 = initial_z0
#     return bpm_pos, orbits
# 
# 
# def _get_orbit_bump(SC, cm_ord, bpm_ord, n_dim, par):  # TODO
#     tmpCMind = np.where(par.RMstruct.CMords[0] == cm_ord)[0]
#     if len(tmpCMind):
#         par.RMstruct.RM = np.delete(par.RMstruct.RM, tmpCMind, 1)  # TODO not nice
#         par.RMstruct.CMords[0] = np.delete(par.RMstruct.CMords[0], tmpCMind)
#     tmpBPMind = np.where(bpm_ord == par.RMstruct.BPMords)[0]
# 
#     R0 = bpm_reading(SC) if par.use_bpm_reading_for_orbit_bump_ref else np.zeros((2, len(par.RMstruct.BPMords)))
#     R0[n_dim, tmpBPMind] += par.BBABPMtarget
#     cm_ords = par.RMstruct.CMords
#     W0 = np.ones((2, len(par.RMstruct.BPMords)))  # TODO weight for SCFedbackRun
#     W0[n_dim, max(1, tmpBPMind - par.orbBumpWindow):(tmpBPMind - 1)] = 0
#     W0[n_dim, (tmpBPMind + 1):min(len(par.RMstruct.BPMords), tmpBPMind + par.orbBumpWindow)] = 0
# 
#     CUR = orbit_trajectory.correct(SC, par.RMstruct.RM, reference=R0, cm_ords=cm_ords, bpm_ords=par.RMstruct.BPMords, eps=1E-6,
#                   target=0, maxsteps=50, scaleDisp=par.RMstruct.scaleDisp, )
#     cm_vec = []
#     factor = np.linspace(-1, 1, par.n_steps)
#     for n_dim in range(2):
#         vec0 = SC.get_cm_setpoints(cm_ords[n_dim], skewness=bool(n_dim))
#         vec1 = CUR.get_cm_setpoints(cm_ords[n_dim], skewness=bool(n_dim))
#         cm_vec.append(vec0 + np.outer(factor, vec0 - vec1))
# 
#     return cm_ords, cm_vec
# 
# 
# def _plot_bba_step(SC, ax, bpm_ind, n_dim):
#     s_pos = at_wrapper.findspos(SC.RING)
#     bpm_readings, all_elements_positions = all_elements_reading(SC)
#     ax.plot(s_pos[SC.ORD.BPM], 1E3 * bpm_readings[n_dim, :len(SC.ORD.BPM)], marker='o')
#     ax.plot(s_pos[SC.ORD.BPM[bpm_ind]], 1E3 * bpm_readings[n_dim, bpm_ind], marker='o', markersize=10, markerfacecolor='k')
#     ax.plot(s_pos, 1E3 * all_elements_positions[n_dim, 0, :, 0, 0], linestyle='-')
#     return ax
