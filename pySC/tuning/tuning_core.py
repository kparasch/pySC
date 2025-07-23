from pydantic import BaseModel, PrivateAttr
from typing import Optional, Union, TYPE_CHECKING
from .response_matrix import ResponseMatrix
from .response_measurements import measure_TrajectoryResponseMatrix, measure_OrbitResponseMatrix
from .trajectory_bba import Trajectory_BBA_Configuration, trajectory_bba, get_mag_s_pos

import numpy as np
from pathlib import Path
import json

if TYPE_CHECKING:
    from ..core.new_simulated_commissioning import SimulatedCommissioning

class Tuning(BaseModel, extra="forbid"):
    HCORR: list[str] = []
    VCORR: list[str] = []
    multipoles: list[str] = []
    bba_magnets: list[str] = []
    trajectory_bba_config: Optional[Trajectory_BBA_Configuration] = None
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
                print(f'Loading {name} RM from file: {rm_path}')
                self.response_matrix[name] = ResponseMatrix.model_validate(json.load(open(rm_path,'r')))
            else:
                if orbit:
                    self.calculate_model_orbit_response_matrix()
                else:
                    self.calculate_model_trajectory_response_matrix(n_turns=n_turns)
        return

    def calculate_model_trajectory_response_matrix(self, n_turns=1, dkick=1e-5, save_as: str = None):
        RM_name = f'trajectory{n_turns}'
        RM = measure_TrajectoryResponseMatrix(self._parent, n_turns=n_turns, dkick=dkick, use_design=True)
        self.response_matrix[RM_name] = ResponseMatrix(RM=RM)
        if save_as is not None:
            json.dump(self.response_matrix[RM_name].model_dump(), open(save_as, 'w'))
        return 

    def calculate_model_orbit_response_matrix(self, dkick=1e-5, save_as: str = None):
        RM_name = 'orbit'
        RM = measure_OrbitResponseMatrix(self._parent, dkick=dkick, use_design=True)
        self.response_matrix[RM_name] = ResponseMatrix(RM=RM)
        if save_as is not None:
            json.dump(self.response_matrix[RM_name].model_dump(), open(save_as, 'w'))
        return 

    def correct_injection(self, n_turns=1, n_reps=1, method='tikhonov', parameter=100, gain=1):
        RM_name = f'trajectory{n_turns}'
        self.fetch_response_matrix(RM_name, orbit=False)
        RM = self.response_matrix[RM_name]

        for _ in range(n_reps):
            trajectory_x, trajectory_y = self._parent.bpm_system.capture_injection(n_turns=n_turns)
            trajectory = np.concat((trajectory_x.flatten(order='F'), trajectory_y.flatten(order='F')))

            trims = RM.solve(trajectory, method=method, parameter=parameter)

            settings = self._parent.magnet_settings
            for control_name, trim in zip(self.CORR, trims):
                setpoint = settings.get(control_name=control_name) - gain*trim
                settings.set(control_name=control_name, setpoint=setpoint)

        trajectory_x, trajectory_y = self._parent.bpm_system.capture_injection(n_turns=n_turns)
        trajectory_x = trajectory_x.flatten('F')
        trajectory_y = trajectory_y.flatten('F')
        rms_x = np.nanstd(trajectory_x) * 1e6
        rms_y = np.nanstd(trajectory_y) * 1e6
        bad_readings = sum(np.isnan(trajectory_x))
        good_turns = (len(trajectory_x) - bad_readings) / len(self._parent.bpm_system.indices)
        print(f'Corrected injection: transmission through {good_turns:.2f}/{n_turns} turns, {rms_x=:.1f} um, {rms_y=:.1f} um.')

        return

    def correct_orbit(self, n_reps=1, method='tikhonov', parameter=100):
        RM_name = 'orbit'
        self.fetch_response_matrix(RM_name, orbit=True)
        RM = self.response_matrix[RM_name]

        for _ in range(n_reps):
            orbit_x, orbit_y = self._parent.bpm_system.capture_orbit()
            orbit = np.concat((orbit_x.flatten(order='F'), orbit_y.flatten(order='F')))

            trims = RM.solve(orbit, method=method, parameter=parameter)

            settings = self._parent.magnet_settings
            for control_name, trim in zip(self.CORR, trims):
                setpoint = settings.get(control_name=control_name) - trim
                settings.set(control_name=control_name, setpoint=setpoint)
        return

    def set_multipole_scale(self, scale: float = 1):
        print(f'Setting "multipoles" to {scale*100:.0f}%')
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

    def bba_to_quad_true_offset(self, bpm_name: str, plane=None) -> Union[float, tuple[float,float]]:
        #assert len(SC.tuning.trajectory_bba_config.config) > 0, 'T'

        SC = self._parent
        bpm_number = SC.bpm_system.bpm_number(name=bpm_name)
        bpm_index = SC.bpm_system.indices[bpm_number]
        bpm_s = SC.lattice.twiss['s'][bpm_index]

        bba_magnets = SC.tuning.bba_magnets
        bba_magnets_s = get_mag_s_pos(SC, bba_magnets)
        bba_magnet_number = np.argmin(np.abs(bba_magnets_s - bpm_s))
        quad = bba_magnets[bba_magnet_number]

        quad_index = SC.magnet_settings.magnets[quad.split('/')[0]].sim_index
        true_offset2 =SC.support_system.get_total_offset(quad_index) - SC.support_system.get_total_offset(bpm_index)
        if plane is None:
           return tuple(true_offset2)
        elif plane == 'H':
           return true_offset2[0]
        elif plane == 'V':
           return true_offset2[1]
        else:
            raise Exception(f'Unknown {plane=}')

    def do_trajectory_bba(self, bpm_names: Optional[list[str]] = None, shots_per_trajectory: int = 1):
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
            offset_x, offset_x_err = trajectory_bba(SC, name, plane='H', shots_per_trajectory=shots_per_trajectory)
            print(f'\t{name=}, bpm_number = {bpm_number}')
            print(f'\t\t new H. offset = {offset_x*1e6:.1f} +- {offset_x_err*1e6:.1f} um, true offset is {true_offset_x*1e6:.1f} um')
            offset_y, offset_y_err = trajectory_bba(SC, name, plane='V', shots_per_trajectory=shots_per_trajectory)
            print(f'\t\t new V. offset = {offset_y*1e6:.1f} +- {offset_y_err*1e6:.1f} um, true offset is {true_offset_y*1e6:.1f} um')

            true_offsets_x[ii] = true_offset_x
            true_offsets_y[ii] = true_offset_y
            offsets_x[ii] = offset_x
            offsets_y[ii] = offset_y

        acc_x = 1e6 * np.nanstd(offsets_x - true_offsets_x)
        acc_y = 1e6 * np.nanstd(offsets_y - true_offsets_y)
        print(f'Trajectory BBA accuracy, H: {acc_x:.1f} um, V: {acc_y:.1f} um')

        for ii, bpm_number in enumerate(bpm_numbers):
            SC.bpm_system.bba_offsets_x[bpm_number] = offsets_x[ii]
            SC.bpm_system.bba_offsets_y[bpm_number] = offsets_y[ii]
        return offsets_x, offsets_y

    def injection_efficiency(self, n_turns: int = 1) -> float:
        SC = self._parent
        bunch = SC.injection.generate_bunch()
        track_data = SC.lattice.track(bunch, indices=SC.bpm_system.indices, n_turns=n_turns, use_design=False)
        transmission = np.sum(~np.isnan(track_data[0]), axis=0) / len(bunch)
        return transmission[-1, :]