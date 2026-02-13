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

    def get_inputs_plane(self, control_names):
        SC = self._parent
        inputs_plane = []
        for corr in control_names:
            control = SC.magnet_settings.controls[corr]
            if type(control.info) is not IndivControl:
                raise NotImplementedError(f'Unsupported control type for {corr} of type {type(control.info).__name__}.')
            if control.info.component == 'B':
                inputs_plane.append('H')
            elif control.info.component == 'A':
                inputs_plane.append('V')
            else:
                raise Exception(f'Unknown component: {control.info.component}')
        return inputs_plane

    def calculate_model_trajectory_response_matrix(self, n_turns=1, dkick=1e-5, save_as: str = None):
        # assumes all bpms are dual plane
        RM_name = f'trajectory{n_turns}'
        SC = self._parent
        input_names = SC.tuning.CORR
        output_names = SC.bpm_system.names * n_turns * 2 # two: one per plane and per turn
        matrix = measure_TrajectoryResponseMatrix(SC, n_turns=n_turns, dkick=dkick, use_design=True)
        inputs_plane  = self.get_inputs_plane(SC.tuning.CORR)

        self.response_matrix[RM_name] = ResponseMatrix(matrix=matrix, output_names=output_names,
                                                       input_names=input_names, inputs_plane=inputs_plane)
        if save_as is not None:
            self.response_matrix[RM_name].to_json(save_as)
        return 

    def calculate_model_orbit_response_matrix(self, dkick=1e-5, save_as: str = None):
        RM_name = 'orbit'
        SC = self._parent
        input_names = SC.tuning.CORR
        output_names = SC.bpm_system.names * 2 # two: one per plane
        matrix = measure_OrbitResponseMatrix(SC, dkick=dkick, use_design=True)
        inputs_plane  = self.get_inputs_plane(SC.tuning.CORR)
        self.response_matrix[RM_name] = ResponseMatrix(matrix=matrix, output_names=output_names,
                                                       input_names=input_names, inputs_plane=inputs_plane)
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

    def correct_injection(self, n_turns=1, n_reps=1, method='tikhonov', parameter=100, gain=1, correct_to_first_turn=False, zerosum=False):
        RM_name = f'trajectory{n_turns}'
        self.fetch_response_matrix(RM_name, orbit=False, n_turns=n_turns)
        response_matrix = self.response_matrix[RM_name]
        response_matrix.bad_outputs = self.bad_outputs_from_bad_bpms(self.bad_bpms, n_turns=n_turns)

        SC = self._parent
        interface = pySCInjectionInterface(SC=SC, n_turns=n_turns)

        for _ in range(n_reps):
            _ = orbit_correction(interface=interface, response_matrix=response_matrix, reference=None,
                                     method=method, parameter=parameter, gain=gain, apply=True)

        trajectory_x, trajectory_y = SC.bpm_system.capture_injection(n_turns=n_turns)
        trajectory_x = trajectory_x.flatten('F')
        trajectory_y = trajectory_y.flatten('F')
        rms_x = np.nanstd(trajectory_x) * 1e6
        rms_y = np.nanstd(trajectory_y) * 1e6
        bad_readings = sum(np.isnan(trajectory_x))
        good_turns = (len(trajectory_x) - bad_readings) / len(SC.bpm_system.indices)
        logger.info(f'Corrected injection: transmission through {good_turns:.2f}/{n_turns} turns, {rms_x=:.1f} um, {rms_y=:.1f} um.')

        return

    def correct_orbit(self, n_reps=1, method='tikhonov', parameter=100, gain=1, zerosum=False):
        RM_name = 'orbit'
        self.fetch_response_matrix(RM_name, orbit=True)
        response_matrix = self.response_matrix[RM_name]
        response_matrix.bad_outputs = self.bad_outputs_from_bad_bpms(self.bad_bpms)

        SC = self._parent
        interface = pySCOrbitInterface(SC=SC)

        for _ in range(n_reps):
            _ = orbit_correction(interface=interface, response_matrix=response_matrix, reference=None,
                                     method=method, parameter=parameter, zerosum=zerosum, gain=gain, apply=True)

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

        if not skip_summary:
            acc_x = 1e6 * np.nanstd(offsets_x - true_offsets_x)
            acc_y = 1e6 * np.nanstd(offsets_y - true_offsets_y)
            logger.info(f'Trajectory BBA accuracy, H: {acc_x:.1f} um, V: {acc_y:.1f} um')

        for ii, bpm_number in enumerate(bpm_numbers):
            SC.bpm_system.bba_offsets_x[bpm_number] = offsets_x[ii]
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

        if not skip_summary:
            acc_x = 1e6 * np.nanstd(offsets_x - true_offsets_x)
            acc_y = 1e6 * np.nanstd(offsets_y - true_offsets_y)
            logger.info(f'Orbit BBA accuracy, H: {acc_x:.1f} um, V: {acc_y:.1f} um')

        for ii, bpm_number in enumerate(bpm_numbers):
            SC.bpm_system.bba_offsets_x[bpm_number] = offsets_x[ii]
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
        for bpm_names_chunk, offsets_x_chunk, offsets_y_chunk in rets:
            for name, offset_x, offset_y in zip(bpm_names_chunk, offsets_x_chunk, offsets_y_chunk):
                ii = bpm_mapping[name]
                offsets_x[ii] = offset_x
                offsets_y[ii] = offset_y
                true_offsets_x[ii], true_offsets_y[ii] = SC.tuning.bba_to_quad_true_offset(bpm_name=name)

        # 4. CLEANUP
        listener.stop()
        logger.handlers.clear()

        acc_x = 1e6 * np.nanstd(offsets_x - true_offsets_x)
        acc_y = 1e6 * np.nanstd(offsets_y - true_offsets_y)
        logger.info(f'Trajectory BBA accuracy, H: {acc_x:.1f} um, V: {acc_y:.1f} um')

        bpm_numbers = [SC.bpm_system.bpm_number(name=name) for name in bpm_names]
        for ii, bpm_number in enumerate(bpm_numbers):
            SC.bpm_system.bba_offsets_x[bpm_number] = offsets_x[ii]
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
        for bpm_names_chunk, offsets_x_chunk, offsets_y_chunk in rets:
            for name, offset_x, offset_y in zip(bpm_names_chunk, offsets_x_chunk, offsets_y_chunk):
                ii = bpm_mapping[name]
                offsets_x[ii] = offset_x
                offsets_y[ii] = offset_y
                true_offsets_x[ii], true_offsets_y[ii] = SC.tuning.bba_to_quad_true_offset(bpm_name=name)

        # 4. CLEANUP
        listener.stop()
        logger.handlers.clear()

        acc_x = 1e6 * np.nanstd(offsets_x - true_offsets_x)
        acc_y = 1e6 * np.nanstd(offsets_y - true_offsets_y)
        logger.info(f'Orbit BBA accuracy, H: {acc_x:.1f} um, V: {acc_y:.1f} um')

        bpm_numbers = [SC.bpm_system.bpm_number(name=name) for name in bpm_names]
        for ii, bpm_number in enumerate(bpm_numbers):
            SC.bpm_system.bba_offsets_x[bpm_number] = offsets_x[ii]
            SC.bpm_system.bba_offsets_y[bpm_number] = offsets_y[ii]
        return offsets_x, offsets_y

    def injection_efficiency(self, n_turns: int = 1, omp_num_threads: Optional[int] = None) -> float:
        SC = self._parent

        if omp_num_threads is not None:
            previous_threads = SC.lattice.omp_num_threads
            SC.lattice.omp_num_threads = omp_num_threads

        bunch = SC.injection.generate_bunch()
        track_data = SC.lattice.track(bunch, indices=SC.bpm_system.indices, n_turns=n_turns, use_design=False)
        transmission = np.sum(~np.isnan(track_data[0]), axis=0) / len(bunch)

        if omp_num_threads is not None:
            SC.lattice.omp_num_threads = previous_threads

        return transmission[-1, :]