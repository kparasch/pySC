from pydantic import BaseModel, PrivateAttr
from typing import Optional, Union, TYPE_CHECKING
from ..apps.response_matrix import ResponseMatrix
from .response_measurements import measure_TrajectoryResponseMatrix, measure_OrbitResponseMatrix, measure_RFFrequencyOrbitResponse
from .trajectory_bba import Trajectory_BBA_Configuration, trajectory_bba, get_mag_s_pos
from .orbit_bba import Orbit_BBA_Configuration, orbit_bba
from .parallel import parallel_tbba_target, parallel_obba_target, get_listener_and_queue
from .tune import Tune
from .chromaticity import Chromaticity
from .c_minus import CMinus
from .rf_tuning import RF_tuning
from ..core.control import IndivControl
from .pySC_interface import pySCInjectionInterface, pySCOrbitInterface
from ..apps import orbit_correction

import numpy as np
import warnings
from pathlib import Path
import logging
from multiprocessing import Process, Queue

if TYPE_CHECKING:
    from ..core.simulated_commissioning import SimulatedCommissioning

logger = logging.getLogger(__name__)

class Tuning(BaseModel, extra="forbid"):
    HCORR: list[str] = []
    VCORR: list[str] = []
    bad_bpms: list[int] = []
    multipoles: list[str] = []

    tune: Tune = Tune() ## TODO: generate config from yaml file
    chromaticity: Chromaticity = Chromaticity() ## TODO: generate config from yaml file
    c_minus: CMinus = CMinus()
    rf: RF_tuning = RF_tuning() ## TODO: generate config from yaml file

    bba_magnets: list[str] = []
    trajectory_bba_config: Optional[Trajectory_BBA_Configuration] = None
    orbit_bba_config: Optional[Orbit_BBA_Configuration] = None
    RM_folder: Optional[str] = None

    _responses: Optional[dict[str, ResponseMatrix]] = PrivateAttr(default={})
    _parent: Optional["SimulatedCommissioning"] = PrivateAttr(default=None)

    @property
    def CORR(self):
        return self.HCORR + self.VCORR

    @property
    def response_matrix(self):
        return self._responses

    def fetch_response_matrix(self, name: str, orbit=True, n_turns=1) -> None:
        """
        fetch in the spirit of "git fetch", it updates the internal list of response matrices to contain the requested one.
        First tries to load it from a file called <self.RM_folder>/<name>.json if self.RM_folder is set.
        If it doesn't exist or if the self.RM_folder path is not set, then it is recalculated.
        """
        if name not in self.response_matrix and self.RM_folder is not None:
            rm_path = Path(self.RM_folder) / Path(name + '.json')
            if rm_path.exists():
                logger.info(f'Loading {name} RM from file: {rm_path}')
                self.response_matrix[name] = ResponseMatrix.from_json(rm_path)
            else:
                if orbit:
                    self.calculate_model_orbit_response_matrix()
                else:
                    self.calculate_model_trajectory_response_matrix(n_turns=n_turns)
        return

    def get_input_planes(self, control_names):
        SC = self._parent
        input_planes = []
        for corr in control_names:
            control = SC.magnet_settings.controls[corr]
            if type(control.info) is not IndivControl:
                raise NotImplementedError(f'Unsupported control type for {corr} of type {type(control.info).__name__}.')
            if control.info.component == 'B':
                input_planes.append('H')
            elif control.info.component == 'A':
                input_planes.append('V')
            else:
                raise Exception(f'Unknown component: {control.info.component}')
        return input_planes

    def calculate_model_trajectory_response_matrix(self, n_turns=1, dkick=1e-5, save_as: str = None):
        # assumes all bpms are dual plane
        RM_name = f'trajectory{n_turns}'
        SC = self._parent
        input_names = SC.tuning.CORR
        output_names = SC.bpm_system.names * n_turns * 2 # two: one per plane and per turn
        matrix = measure_TrajectoryResponseMatrix(SC, n_turns=n_turns, dkick=dkick, use_design=True)
        input_planes  = self.get_input_planes(SC.tuning.CORR)

        self.response_matrix[RM_name] = ResponseMatrix(matrix=matrix, output_names=output_names,
                                                       input_names=input_names, input_planes=input_planes)
        if save_as is not None:
            self.response_matrix[RM_name].to_json(save_as)
        return 

    def calculate_model_orbit_response_matrix(self, dkick=1e-5, save_as: str = None):
        RM_name = 'orbit'
        SC = self._parent
        input_names = SC.tuning.CORR
        output_names = SC.bpm_system.names * 2 # two: one per plane
        matrix = measure_OrbitResponseMatrix(SC, dkick=dkick, use_design=True)
        input_planes  = self.get_input_planes(SC.tuning.CORR)
        self.response_matrix[RM_name] = ResponseMatrix(matrix=matrix, output_names=output_names,
                                                       input_names=input_names, input_planes=input_planes)
        if save_as is not None:
            self.response_matrix[RM_name].to_json(save_as)
        return 

    def bad_outputs_from_bad_bpms(self, bad_bpms: list[int], n_turns: int = 1) -> list[int]:
        n_bpms = len(self._parent.bpm_system.indices)
        bad_outputs = []
        for plane in [0, 1]:
            for turn in range(n_turns):
                for bpm in bad_bpms:
                    bad_outputs.append(bpm + turn * n_bpms + plane * n_turns * n_bpms)
        return bad_outputs

    def wiggle_last_corrector(self, max_steps: int = 100, max_sp: float = 500e-6) -> None:
        SC = self._parent
        def first_turn_transmission(SC):
            x, _ = SC.bpm_system.capture_injection()
            bad_readings = sum(np.isnan(x))
            good_frac = (len(x) - bad_readings) / len(SC.bpm_system.indices)
            last_good_bpm = np.where(~np.isnan(x))[0][-1]
            last_good_bpm_index = SC.bpm_system.indices[last_good_bpm]
            return good_frac, last_good_bpm_index

        initial_transmission, last_good_bpm_index = first_turn_transmission(SC)
        if initial_transmission < 1.:
            for corr in SC.tuning.HCORR:
                hcor_name = corr.split('/')[0]
                hcor_index = SC.magnet_settings.magnets[hcor_name].sim_index
                if hcor_index < last_good_bpm_index:
                    last_hcor = corr
            for corr in SC.tuning.VCORR:
                vcor_name = corr.split('/')[0]
                vcor_index = SC.magnet_settings.magnets[vcor_name].sim_index
                if vcor_index < last_good_bpm_index:
                    last_vcor = corr

            for _ in range(max_steps):
                SC.magnet_settings.set(last_hcor, SC.rng.uniform(-max_sp, max_sp))
                SC.magnet_settings.set(last_vcor, SC.rng.uniform(-max_sp, max_sp))
                transmission, _ = first_turn_transmission(SC)
                if transmission > initial_transmission:
                    logger.info(f"Wiggling improved first-turn transmission from {initial_transmission} to {transmission}.")
                    return
            logger.info("Wiggling failed. Reached maximum number of steps.")
        else:
            logger.info("No need to wiggle, full transmission through first-turn.")

        return

    def correct_injection(self, n_turns=1, n_reps=1, method='tikhonov', parameter=100, gain=1, correct_to_first_turn=False, virtual=False):
        RM_name = f'trajectory{n_turns}'
        self.fetch_response_matrix(RM_name, orbit=False, n_turns=n_turns)
        response_matrix = self.response_matrix[RM_name]
        response_matrix.bad_outputs = self.bad_outputs_from_bad_bpms(self.bad_bpms, n_turns=n_turns)

        SC = self._parent
        interface = pySCInjectionInterface(SC=SC, n_turns=n_turns)

        for _ in range(n_reps):
            _ = orbit_correction(interface=interface, response_matrix=response_matrix, reference=None,
                                     method=method, parameter=parameter, gain=gain, virtual=virtual, apply=True)

        trajectory_x, trajectory_y = SC.bpm_system.capture_injection(n_turns=n_turns)
        trajectory_x = trajectory_x.flatten('F')
        trajectory_y = trajectory_y.flatten('F')
        rms_x = np.nanstd(trajectory_x) * 1e6
        rms_y = np.nanstd(trajectory_y) * 1e6
        bad_readings = sum(np.isnan(trajectory_x))
        good_turns = (len(trajectory_x) - bad_readings) / len(SC.bpm_system.indices)
        logger.info(f'Corrected injection: transmission through {good_turns:.2f}/{n_turns} turns, {rms_x=:.1f} um, {rms_y=:.1f} um.')

        return

    def correct_orbit(self, n_reps=1, method='tikhonov', parameter=100, gain=1, virtual=False):
        RM_name = 'orbit'
        self.fetch_response_matrix(RM_name, orbit=True)
        response_matrix = self.response_matrix[RM_name]
        response_matrix.bad_outputs = self.bad_outputs_from_bad_bpms(self.bad_bpms)

        SC = self._parent
        interface = pySCOrbitInterface(SC=SC)

        for _ in range(n_reps):
            _ = orbit_correction(interface=interface, response_matrix=response_matrix, reference=None,
                                     method=method, parameter=parameter, virtual=virtual, gain=gain, apply=True)

        orbit_x, orbit_y = SC.bpm_system.capture_orbit()
        rms_x = np.nanstd(orbit_x) * 1e6
        rms_y = np.nanstd(orbit_y) * 1e6
        logger.info(f'Corrected orbit: {rms_x=:.1f} um, {rms_y=:.1f} um.')
        return

    # def correct_pseudo_orbit_at_injection(self, n_turns=1, n_reps=1, method='tikhonov', parameter=100, gain=1, zerosum=False):
    #     RM_name = 'orbit'
    #     self.fetch_response_matrix(RM_name, orbit=True)
    #     RM = self.response_matrix[RM_name]
    #     RM.bad_outputs = self.bad_outputs_from_bad_bpms(self.bad_bpms)

    #     for _ in range(n_reps):
    #         trajectory_x, trajectory_y = self._parent.bpm_system.capture_injection(n_turns=n_turns)
    #         pseudo_orbit_x = np.nanmean(trajectory_x, axis=1)
    #         pseudo_orbit_y = np.nanmean(trajectory_y, axis=1)
    #         pseudo_orbit = np.concat((pseudo_orbit_x, pseudo_orbit_y))

    #         trims = RM.solve(pseudo_orbit, method=method, parameter=parameter, zerosum=zerosum)

    #         settings = self._parent.magnet_settings
    #         for control_name, trim in zip(self.CORR, trims):
    #             setpoint = settings.get(control_name=control_name) - gain * trim
    #             settings.set(control_name=control_name, setpoint=setpoint)

    #     trajectory_x, trajectory_y = self._parent.bpm_system.capture_injection(n_turns=n_turns)
    #     trajectory_x = trajectory_x.flatten('F')
    #     trajectory_y = trajectory_y.flatten('F')
    #     rms_x = np.nanstd(trajectory_x) * 1e6
    #     rms_y = np.nanstd(trajectory_y) * 1e6
    #     bad_readings = sum(np.isnan(trajectory_x))
    #     good_turns = (len(trajectory_x) - bad_readings) / len(self._parent.bpm_system.indices)
    #     logger.info(f'Corrected injection: transmission through {good_turns:.2f}/{n_turns} turns, {rms_x=:.1f} um, {rms_y=:.1f} um.')

    #     return

    def fit_dispersive_orbit(self):
        SC = self._parent
        response = measure_RFFrequencyOrbitResponse(SC=SC, use_design=True)

        x,y = SC.bpm_system.capture_orbit(bba=False, subtract_reference=False, use_design=False)
        xy =  np.concat((x.flatten(order='F'), y.flatten(order='F')))

        return np.dot(xy, response) / np.dot(response, response)

    def set_multipole_scale(self, scale: float = 1):
        logger.info(f'Setting "multipoles" to {scale*100:.0f}%')
        for control_name in self.multipoles:
            setpoint = self._parent.design_magnet_settings.get(control_name)
            self._parent.magnet_settings.set(control_name, scale*setpoint)

    def reset_to_design(self):
        for control_name in self._parent.magnet_settings.controls.keys():
            setpoint = self._parent.design_magnet_settings.get(control_name)
            self._parent.magnet_settings.set(control_name, setpoint)

    def generate_trajectory_bba_config(self, max_dx_at_bpm: float = 1e-3, 
                                       max_modulation: float = 0.2e-3,
                                       n_downstream_bpms: int = 50, 
                                       max_ncorr_index: int = 10) -> None:
        config = Trajectory_BBA_Configuration.generate_config(SC=self._parent,
                                                              max_dx_at_bpm=max_dx_at_bpm,
                                                              max_modulation=max_modulation,
                                                              n_downstream_bpms=n_downstream_bpms,
                                                              max_ncorr_index=max_ncorr_index)
        self.trajectory_bba_config = config
        return

    def generate_orbit_bba_config(self, max_dx_at_bpm: float = 0.3e-3, 
                                       max_modulation: float = 20e-6) -> None:
        config = Orbit_BBA_Configuration.generate_config(SC=self._parent,
                                                         max_dx_at_bpm=max_dx_at_bpm,
                                                         max_modulation=max_modulation)
        self.orbit_bba_config = config
        return

    def bba_to_quad_true_offset(self, bpm_name: str, plane=None) -> Union[float, tuple[float,float]]:
        #assert len(SC.tuning.trajectory_bba_config.config) > 0, 'T'

        SC = self._parent
        bpm_number = SC.bpm_system.bpm_number(name=bpm_name)
        bpm_index = SC.bpm_system.indices[bpm_number]
        bpm_s = SC.lattice.twiss['s'][bpm_index]

        bba_magnet_controls = SC.tuning.bba_magnets
        bba_magnets_s = get_mag_s_pos(SC, bba_magnet_controls)
        bba_magnet_number = np.argmin(np.abs(bba_magnets_s - bpm_s))
        quad = bba_magnet_controls[bba_magnet_number]
        bba_control_info = SC.magnet_settings.controls[quad].info
        assert type(bba_control_info) is IndivControl
        bba_magnet_name = bba_control_info.magnet_name

        quad_index = SC.magnet_settings.magnets[bba_magnet_name].sim_index
        true_offset2 = SC.support_system.get_total_offset(quad_index) - SC.support_system.get_total_offset(bpm_index)
        if plane is None:
           return tuple(true_offset2)
        elif plane == 'H':
           return true_offset2[0]
        elif plane == 'V':
           return true_offset2[1]
        else:
            raise Exception(f'Unknown {plane=}')

    def fake_align_bpms(self, bpm_names: Optional[list[str]] = None, rms_offset: float = 0.):
        SC = self._parent
        if bpm_names is None:
            bpm_names = SC.bpm_system.names

        for bpm_name in bpm_names:
            bpm_number = SC.bpm_system.bpm_number(name=bpm_name)
            bba_x, bba_y = self.bba_to_quad_true_offset(bpm_name=bpm_name)
            if rms_offset > 0:
                bba_x += SC.rng.normal_trunc(loc=0, scale=rms_offset)
                bba_y += SC.rng.normal_trunc(loc=0, scale=rms_offset)
            self._parent.bpm_system.bba_offsets_x[bpm_number] = bba_x
            self._parent.bpm_system.bba_offsets_y[bpm_number] = bba_y

    def do_trajectory_bba(self, bpm_names: Optional[list[str]] = None, shots_per_trajectory: int = 1, skip_summary: bool = False,
                          n_corr_steps: int = 5):
        SC = self._parent
        if bpm_names is None:
            bpm_names = SC.bpm_system.names

        n_bpm = len(bpm_names)
        bpm_numbers = [SC.bpm_system.bpm_number(name=name) for name in bpm_names]
        offsets_x = np.zeros(n_bpm)
        offsets_y = np.zeros(n_bpm)
        true_offsets_x = np.zeros(n_bpm)
        true_offsets_y = np.zeros(n_bpm)
        reconstructed_offsets_x = np.zeros(n_bpm)
        reconstructed_offsets_y = np.zeros(n_bpm)
        for ii, name in enumerate(bpm_names):
            true_offset_x, true_offset_y = self.bba_to_quad_true_offset(bpm_name=name)
            bpm_number = SC.bpm_system.bpm_number(name=name)
            offset_x, offset_x_err = trajectory_bba(SC, name, plane='H', shots_per_trajectory=shots_per_trajectory, n_corr_steps=n_corr_steps)
            logger.info(f'T. BBA: Name={name}, number={bpm_number}, new H. offset = {offset_x*1e6:.1f} +- {offset_x_err*1e6:.1f} um, true is {true_offset_x*1e6:.1f} um')
            offset_y, offset_y_err = trajectory_bba(SC, name, plane='V', shots_per_trajectory=shots_per_trajectory, n_corr_steps=n_corr_steps)
            logger.info(f'T. BBA: Name={name}, number={bpm_number}, new V. offset = {offset_y*1e6:.1f} +- {offset_y_err*1e6:.1f} um, true is {true_offset_y*1e6:.1f} um')

            true_offsets_x[ii] = true_offset_x
            true_offsets_y[ii] = true_offset_y
            offsets_x[ii] = offset_x
            offsets_y[ii] = offset_y
            reconstructed_offsets_x[ii], reconstructed_offsets_y[ii] = SC.bpm_system.reconstruct_true_orbit(name=name, x=offset_x, y=offset_y)

        if not skip_summary:
            acc_x = 1e6 * np.nanstd(reconstructed_offsets_x - true_offsets_x)
            acc_y = 1e6 * np.nanstd(reconstructed_offsets_y - true_offsets_y)
            logger.info(f'Trajectory BBA accuracy, H: {acc_x:.1f} um, V: {acc_y:.1f} um')

        for ii, bpm_number in enumerate(bpm_numbers):
            if offsets_x[ii] == offsets_x[ii]: #is not nan
                SC.bpm_system.bba_offsets_x[bpm_number] = offsets_x[ii]
            if offsets_y[ii] == offsets_y[ii]: #is not nan
                SC.bpm_system.bba_offsets_y[bpm_number] = offsets_y[ii]
        return offsets_x, offsets_y

    def do_orbit_bba(self, bpm_names: Optional[list[str]] = None, shots_per_orbit: int = 1, skip_summary: bool = False, n_corr_steps: int = 5):
        SC = self._parent
        if bpm_names is None:
            bpm_names = SC.bpm_system.names

        n_bpm = len(bpm_names)
        bpm_numbers = [SC.bpm_system.bpm_number(name=name) for name in bpm_names]
        offsets_x = np.zeros(n_bpm)
        offsets_y = np.zeros(n_bpm)
        true_offsets_x = np.zeros(n_bpm)
        true_offsets_y = np.zeros(n_bpm)
        reconstructed_offsets_x = np.zeros(n_bpm)
        reconstructed_offsets_y = np.zeros(n_bpm)
        for ii, name in enumerate(bpm_names):
            true_offset_x, true_offset_y = self.bba_to_quad_true_offset(bpm_name=name)
            bpm_number = SC.bpm_system.bpm_number(name=name)
            offset_x, offset_x_err = orbit_bba(SC, name, plane='H', shots_per_orbit=shots_per_orbit, n_corr_steps=n_corr_steps)
            logger.info(f'O. BBA: Name={name}, number={bpm_number}, new H. offset = {offset_x*1e6:.1f} +- {offset_x_err*1e6:.1f} um, true is {true_offset_x*1e6:.1f} um')
            offset_y, offset_y_err = orbit_bba(SC, name, plane='V', shots_per_orbit=shots_per_orbit, n_corr_steps=n_corr_steps)
            logger.info(f'O. BBA: Name={name}, number={bpm_number}, new V. offset = {offset_y*1e6:.1f} +- {offset_y_err*1e6:.1f} um, true is {true_offset_y*1e6:.1f} um')

            true_offsets_x[ii] = true_offset_x
            true_offsets_y[ii] = true_offset_y
            offsets_x[ii] = offset_x
            offsets_y[ii] = offset_y
            reconstructed_offsets_x[ii], reconstructed_offsets_y[ii] = SC.bpm_system.reconstruct_true_orbit(name=name, x=offset_x, y=offset_y)

        if not skip_summary:
            acc_x = 1e6 * np.nanstd(reconstructed_offsets_x - true_offsets_x)
            acc_y = 1e6 * np.nanstd(reconstructed_offsets_y - true_offsets_y)
            logger.info(f'Orbit BBA accuracy, H: {acc_x:.1f} um, V: {acc_y:.1f} um')

        for ii, bpm_number in enumerate(bpm_numbers):
            if offsets_x[ii] == offsets_x[ii]: #is not nan
                SC.bpm_system.bba_offsets_x[bpm_number] = offsets_x[ii]
            if offsets_y[ii] == offsets_y[ii]: #is not nan
                SC.bpm_system.bba_offsets_y[bpm_number] = offsets_y[ii]
        return offsets_x, offsets_y

    def do_parallel_trajectory_bba(self, bpm_names: Optional[list[str]] = None, shots_per_trajectory: int = 1, omp_num_threads: int = 2,
                                   n_corr_steps: int = 5):
        SC = self._parent
        if bpm_names is None:
            bpm_names = SC.bpm_system.names

        logger.info(f'Running parallel trajectory BBA with {omp_num_threads} processes.')
        n_bpm = len(bpm_names)
        bpm_mapping = {name: ii for ii, name in enumerate(bpm_names)}
        # 1. SPLIT bpm_names into n_processes chunks
        bpm_names_chunks = [bpm_names[i::omp_num_threads] for i in range(omp_num_threads)]

        SC_model = SC.model_dump()
        SC_class = SC.__class__

        queue = Queue()
        listener, log_queue = get_listener_and_queue(logger)
        processes = []
        listener.start()
        for num in range(omp_num_threads):
            args = (SC_model, SC_class, bpm_names_chunks[num], shots_per_trajectory, n_corr_steps, queue, log_queue)
            p = Process(target=parallel_tbba_target, args=args)
            processes.append(p)
            p.start()

        # 2. RUN
        rets = []
        for p in processes:
            ret = queue.get()  
            rets.append(ret)
        for p in processes:
            p.join()

        # 3. GATHER
        offsets_x = np.zeros(n_bpm)
        offsets_y = np.zeros(n_bpm)
        true_offsets_x = np.zeros(n_bpm)
        true_offsets_y = np.zeros(n_bpm)
        reconstructed_offsets_x = np.zeros(n_bpm)
        reconstructed_offsets_y = np.zeros(n_bpm)
        for bpm_names_chunk, offsets_x_chunk, offsets_y_chunk in rets:
            for name, offset_x, offset_y in zip(bpm_names_chunk, offsets_x_chunk, offsets_y_chunk):
                ii = bpm_mapping[name]
                offsets_x[ii] = offset_x
                offsets_y[ii] = offset_y
                reconstructed_offsets_x[ii], reconstructed_offsets_y[ii] = SC.bpm_system.reconstruct_true_orbit(name=name, x=offset_x, y=offset_y)
                true_offsets_x[ii], true_offsets_y[ii] = SC.tuning.bba_to_quad_true_offset(bpm_name=name)

        # 4. CLEANUP
        listener.stop()
        logger.handlers.clear()

        acc_x = 1e6 * np.nanstd(reconstructed_offsets_x - true_offsets_x)
        acc_y = 1e6 * np.nanstd(reconstructed_offsets_y - true_offsets_y)
        logger.info(f'Trajectory BBA accuracy, H: {acc_x:.1f} um, V: {acc_y:.1f} um')

        bpm_numbers = [SC.bpm_system.bpm_number(name=name) for name in bpm_names]
        for ii, bpm_number in enumerate(bpm_numbers):
            if offsets_x[ii] == offsets_x[ii]: #is not nan
                SC.bpm_system.bba_offsets_x[bpm_number] = offsets_x[ii]
            if offsets_y[ii] == offsets_y[ii]: #is not nan
                SC.bpm_system.bba_offsets_y[bpm_number] = offsets_y[ii]
        return offsets_x, offsets_y

    def do_parallel_orbit_bba(self, bpm_names: Optional[list[str]] = None, shots_per_orbit: int = 1, omp_num_threads: int = 2, n_corr_steps: int = 5):
        SC = self._parent
        if bpm_names is None:
            bpm_names = SC.bpm_system.names

        logger.info(f'Running parallel orbit BBA with {omp_num_threads} processes.')
        n_bpm = len(bpm_names)
        bpm_mapping = {name: ii for ii, name in enumerate(bpm_names)}
        # 1. SPLIT bpm_names into n_processes chunks
        bpm_names_chunks = [bpm_names[i::omp_num_threads] for i in range(omp_num_threads)]

        SC_model = SC.model_dump()
        SC_class = SC.__class__

        queue = Queue()
        listener, log_queue = get_listener_and_queue(logger)
        processes = []
        listener.start()
        for num in range(omp_num_threads):
            args = (SC_model, SC_class, bpm_names_chunks[num], shots_per_orbit, n_corr_steps, queue, log_queue)
            p = Process(target=parallel_obba_target, args=args)
            processes.append(p)
            p.start()

        # 2. RUN
        rets = []
        for p in processes:
            ret = queue.get()  
            rets.append(ret)
        for p in processes:
            p.join()

        # 3. GATHER
        offsets_x = np.zeros(n_bpm)
        offsets_y = np.zeros(n_bpm)
        true_offsets_x = np.zeros(n_bpm)
        true_offsets_y = np.zeros(n_bpm)
        reconstructed_offsets_x = np.zeros(n_bpm)
        reconstructed_offsets_y = np.zeros(n_bpm)
        for bpm_names_chunk, offsets_x_chunk, offsets_y_chunk in rets:
            for name, offset_x, offset_y in zip(bpm_names_chunk, offsets_x_chunk, offsets_y_chunk):
                ii = bpm_mapping[name]
                offsets_x[ii] = offset_x
                offsets_y[ii] = offset_y
                reconstructed_offsets_x[ii], reconstructed_offsets_y[ii] = SC.bpm_system.reconstruct_true_orbit(name=name, x=offset_x, y=offset_y)
                true_offsets_x[ii], true_offsets_y[ii] = SC.tuning.bba_to_quad_true_offset(bpm_name=name)

        # 4. CLEANUP
        listener.stop()
        logger.handlers.clear()

        acc_x = 1e6 * np.nanstd(reconstructed_offsets_x - true_offsets_x)
        acc_y = 1e6 * np.nanstd(reconstructed_offsets_y - true_offsets_y)
        logger.info(f'Orbit BBA accuracy, H: {acc_x:.1f} um, V: {acc_y:.1f} um')

        bpm_numbers = [SC.bpm_system.bpm_number(name=name) for name in bpm_names]
        for ii, bpm_number in enumerate(bpm_numbers):
            if offsets_x[ii] == offsets_x[ii]: #is not nan
                SC.bpm_system.bba_offsets_x[bpm_number] = offsets_x[ii]
            if offsets_y[ii] == offsets_y[ii]: #is not nan
                SC.bpm_system.bba_offsets_y[bpm_number] = offsets_y[ii]
        return offsets_x, offsets_y

    def tune_scan(self, dqx_range: np.ndarray, dqy_range: np.ndarray,
                  n_turns: int = 50, target: float = 0.8,
                  full_scan: bool = False,
                  omp_num_threads: Optional[int] = None) -> tuple[float, float, np.ndarray, int]:
        """Spiral grid scan of tune deltas for beam survival.

        Scans tune knob setpoints in a spiral pattern starting from the
        center of the grid, looking for settings that achieve the target
        transmission.  Uses the registered tune knobs (see
        ``create_tune_knobs()``) and ``injection_efficiency()`` for tracking.

        Args:
            dqx_range: 1-D array of horizontal tune deltas.
            dqy_range: 1-D array of vertical tune deltas.
            n_turns: Number of turns for tracking (default 50).
            target: Target final-turn transmission fraction (default 0.8).
            full_scan: If True, scan entire grid even after target is reached.
            omp_num_threads: Optional thread count forwarded to tracking.

        Returns:
            (best_dqx, best_dqy, survival_map, error) where:
            - best_dqx: Best horizontal tune delta.
            - best_dqy: Best vertical tune delta.
            - survival_map: 2-D array of final-turn transmissions.
            - error: 0 = target reached, 1 = improved but not target,
              2 = no transmission.
        """
        SC = self._parent

        # Assert knobs are registered
        assert self.tune.knob_qx in SC.magnet_settings.controls, \
            f"Knob '{self.tune.knob_qx}' not registered — call create_tune_knobs() first"
        assert self.tune.knob_qy in SC.magnet_settings.controls, \
            f"Knob '{self.tune.knob_qy}' not registered — call create_tune_knobs() first"

        # Save initial knob setpoints
        initial_qx = SC.magnet_settings.get(self.tune.knob_qx)
        initial_qy = SC.magnet_settings.get(self.tune.knob_qy)

        n1, n2 = len(dqx_range), len(dqy_range)

        # Allocate output
        transmission_map = np.full((n1, n2), np.nan)
        turns_map = np.full((n1, n2), np.nan)

        # Generate spiral scan order (center-first)
        n_max = max(n1, n2)
        spiral_indices = self._spiral_order(n_max)

        best_dqx = float(dqx_range[n1 // 2])
        best_dqy = float(dqy_range[n2 // 2])
        best_trans = 0.0
        best_turns = 0
        scan_order = []

        for q1, q2 in spiral_indices:
            if q1 >= n1 or q2 >= n2:
                continue

            # Set tune knobs (absolute, not cumulative)
            SC.magnet_settings.set(self.tune.knob_qx, initial_qx + dqx_range[q1])
            SC.magnet_settings.set(self.tune.knob_qy, initial_qy + dqy_range[q2])

            survival = self.injection_efficiency(n_turns=n_turns, omp_num_threads=omp_num_threads)
            final_trans = float(survival[-1, -1]) if survival.ndim > 1 else float(survival[-1])
            #survival_map[q1, q2] = survival

            # Find max turns with any survival
            turn_survival = np.mean(survival, axis=0) if survival.ndim > 1 else survival
            max_turns_achieved = np.sum(turn_survival > 0)

            transmission_map[q1, q2] = final_trans
            turns_map[q1, q2] = max_turns_achieved
            scan_order.append((q1, q2))

            if final_trans > best_trans or (final_trans == best_trans and max_turns_achieved > best_turns):
                best_trans = final_trans
                best_turns = max_turns_achieved
                best_dqx = float(dqx_range[q1])
                best_dqy = float(dqy_range[q2])

            # Early termination
            if not full_scan and final_trans >= target:
                logger.info(f'tune_scan: target reached at dqx={dqx_range[q1]:.3f}, dqy={dqy_range[q2]:.3f}, '
                           f'transmission={final_trans:.2%}')
                return best_dqx, best_dqy, transmission_map, 0

        # Apply best deltas at end
        SC.magnet_settings.set(self.tune.knob_qx, initial_qx + best_dqx)
        SC.magnet_settings.set(self.tune.knob_qy, initial_qy + best_dqy)

        if best_trans == 0.0:
            logger.info('tune_scan: no transmission at all')
            return best_dqx, best_dqy, transmission_map, 2

        if best_trans >= target:
            logger.info(f'tune_scan: target reached (full scan), transmission={best_trans:.2%}')
            return best_dqx, best_dqy, transmission_map, 0

        logger.info(f'tune_scan: best transmission={best_trans:.2%} (target={target:.2%})')
        return best_dqx, best_dqy, transmission_map, 1

    @staticmethod
    def _spiral_order(n: int) -> list[tuple[int, int]]:
        """Generate spiral scan indices for an n x n grid, starting from center."""
        x = (n - 1) // 2
        y = (n - 1) // 2
        result = [(x, y)]

        # directions: right, down, left, up
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        step_size = 1
        while len(result) < n * n:
            for d in range(4):
                dx, dy = directions[d]
                steps = step_size

                for _ in range(steps):
                    x += dx
                    y += dy
                    if 0 <= x < n and 0 <= y < n:
                        result.append((x, y))
                        if len(result) == n * n:
                            return result

                # increase step size after moving horizontally (every 2 directions)
                if d % 2 == 1:
                    step_size += 1

        return result

    def synch_energy_correction(self, freq_range: tuple[float, float] = (-1e3, 1e3),
                                n_steps: int = 15, n_turns: int = 150,
                                min_turns: int = 50) -> tuple[float, int]:
        """Calculate beam-based RF frequency correction.

        Ports SCsynchEnergyCorrection from MATLAB SC toolkit. Scans RF frequency,
        measures mean TBT horizontal BPM shift, fits a line to slope vs frequency,
        and returns the zero-crossing as the correction.

        Args:
            freq_range: (min, max) frequency offset range in Hz.
            n_steps: Number of frequency steps to evaluate.
            n_turns: Number of turns to track at each step.
            min_turns: Minimum turns with beam survival to include a measurement.

        Returns:
            (delta_f, error) where:
            - delta_f: Frequency correction to add to cavity frequency [Hz]
            - error: 0=success, 1=no transmission, 2=NaN result
        """
        SC = self._parent
        interface = pySCOrbitInterface(SC=SC)

        f_test_vec = np.linspace(freq_range[0], freq_range[1], n_steps)
        bpm_shift = np.full(n_steps, np.nan)

        # Save original frequency
        original_freq = interface.get_rf_main_frequency()

        for i, df in enumerate(f_test_vec):
            # Temporarily change RF frequency
            interface.set_rf_main_frequency(original_freq + df)

            # Get turn-by-turn BPM readings
            x, y = SC.bpm_system.capture_injection(n_turns=n_turns)
            # x has shape [n_bpm, n_turns]

            if x.ndim < 2:
                continue

            # Compute mean TBT difference from first turn
            # BB[i,j] = x reading at BPM i, turn j
            BB = x  # [n_bpm, n_turns]
            with warnings.catch_warnings(): # suppress RuntimeWarning: Mean of empty slice
                warnings.simplefilter("ignore", category=RuntimeWarning)
                tbt_de = np.nanmean(BB - BB[:, 0:1], axis=0)  # [n_turns]

            # Fit line: slope of energy shift vs turn number
            turns = np.arange(1, n_turns + 1, dtype=float)
            valid = ~np.isnan(tbt_de)
            if np.sum(valid) < min_turns:
                logger.info(f'synch_energy_correction: frequency_shift={df:.2f} Hz, measurement invalid.')
                continue

            x_fit = turns[valid]
            y_fit = tbt_de[valid]
            # Least-squares slope: x \ y
            bpm_shift[i] = np.dot(x_fit, y_fit) / np.dot(x_fit, x_fit)
            logger.info(f'synch_energy_correction: frequency_shift={df:.2f} Hz, bpm_shift={1e6*bpm_shift[i]:.3f} um')

        # Restore original frequency
        interface.set_rf_main_frequency(original_freq)

        # Fit line to slope vs frequency
        valid = ~np.isnan(bpm_shift)
        if np.sum(valid) == 0:
            logger.info('synch_energy_correction: no transmission at any frequency')
            return 0.0, 1

        x_fit = f_test_vec[valid]
        y_fit = bpm_shift[valid]
        p = np.polyfit(x_fit, y_fit, 1)

        # Zero crossing
        delta_f = -p[1] / p[0]

        if np.isnan(delta_f):
            logger.info('synch_energy_correction: NaN correction')
            return 0.0, 2

        logger.info(f'synch_energy_correction: correction = {delta_f:.1f} Hz')
        return delta_f, 0

    def injection_efficiency(self, n_turns: int = 1, omp_num_threads: Optional[int] = None) -> np.ndarray[float]:
        SC = self._parent

        if omp_num_threads is not None:
            previous_threads = SC.lattice.omp_num_threads
            SC.lattice.omp_num_threads = omp_num_threads

        bunch = SC.injection.generate_bunch()
        _, transmission = SC.lattice.track_mean(bunch, indices=SC.bpm_system.indices, n_turns=n_turns, use_design=False)

        if omp_num_threads is not None:
            SC.lattice.omp_num_threads = previous_threads

        return transmission

    def correct_orbit_with_dispersion(self, alpha_sequence=None,
                                      n_reps=1, method='tikhonov', gain=1.0, virtual=False):
        SC = self._parent

        # 1. Fetch and configure response matrix
        self.fetch_response_matrix('orbit')
        response_matrix = self.response_matrix['orbit']
        response_matrix.bad_outputs = self.bad_outputs_from_bad_bpms(self.bad_bpms)

        # 2. Add dispersion to response matrix
        dispersion = measure_RFFrequencyOrbitResponse(SC, use_design=True)
        response_matrix.set_rf_response(dispersion)

        # 3. Default alpha sequence
        if alpha_sequence is None:
            alpha_sequence = [30, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        # 4. Create interface
        interface = pySCOrbitInterface(SC=SC)

        # 5. Progressive correction
        orbit_x, orbit_y = SC.bpm_system.capture_orbit()
        best_rms = np.sqrt(np.nanmean(orbit_x**2) + np.nanmean(orbit_y**2))
        logger.info(f'Orbit+dispersion correction: initial RMS = {best_rms*1e6:.1f} um')

        for alpha in alpha_sequence:
            saved = SC.magnet_settings.get_many(self.CORR)
            saved_rf = interface.get_rf_main_frequency()

            for _ in range(n_reps):
                orbit_correction(interface=interface, response_matrix=response_matrix,
                                 rf=True, method=method, parameter=alpha, gain=gain,
                                 virtual=virtual, apply=True)

            orbit_x, orbit_y = SC.bpm_system.capture_orbit()
            new_rms = np.sqrt(np.nanmean(orbit_x**2) + np.nanmean(orbit_y**2))

            if new_rms < best_rms:
                best_rms = new_rms
                logger.info(f'  alpha={alpha}: RMS improved to {new_rms*1e6:.1f} um')
            else:
                SC.magnet_settings.set_many(saved)
                interface.set_rf_main_frequency(saved_rf)
                logger.info(f'  alpha={alpha}: RMS worsened ({new_rms*1e6:.1f} um), reverting')
                break

        logger.info(f'Orbit+dispersion correction: final RMS = {best_rms*1e6:.1f} um')
