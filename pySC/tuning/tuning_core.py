from pydantic import BaseModel, PrivateAttr
from typing import Optional, TYPE_CHECKING
from .response_matrix import ResponseMatrix
from .model_response import measure_TrajectoryResponseMatrix, measure_OrbitResponseMatrix

import numpy as np
from pathlib import Path
import json

if TYPE_CHECKING:
    from ..core.new_simulated_commissioning import SimulatedCommissioning

class Tuning(BaseModel, extra="forbid"):
    HCORR: list[str] = []
    VCORR: list[str] = []
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

    def calculate_model_trajectory_response_matrix(self, n_turns=1, dkick=1e-5):
        RM_name = f'trajectory{n_turns}'
        RM = measure_TrajectoryResponseMatrix(self._parent, n_turns=1, dkick=dkick, use_design=True)
        self.response_matrix[RM_name] = ResponseMatrix(RM=RM)
        return 

    def calculate_model_orbit_response_matrix(self, n_turns=1, dkick=1e-5):
        RM_name = f'trajectory{n_turns}'
        RM = measure_OrbitResponseMatrix(self._parent, dkick=dkick, use_design=True)
        self.response_matrix[RM_name] = ResponseMatrix(RM=RM)
        return 

    def correct_injection(self, n_turns=1, n_reps=1, method='tikhonov', parameter=100):
        RM_name = f'trajectory{n_turns}'
        self.fetch_response_matrix(RM_name=RM_name, orbit=False)

        RM = self.response_matrix[RM_name]

        for _ in range(n_reps):
            trajectory_x, trajectory_y = self._parent.bpm_system.capture_injection(n_turns=n_turns)
            trajectory = np.concat((trajectory_x.flatten(order='F'), trajectory_y.flatten(order='F')))

            trims = RM.solve(trajectory, method=method, parameter=parameter)

            settings = self._parent.magnet_settings
            for control_name, trim in zip(self.CORR, trims):
                setpoint = settings.get(control_name=control_name) - trim
                settings.set(control_name=control_name, setpoint=setpoint)
        return

    def correct_orbit(self, n_reps=1, method='tikhonov', parameter=100):
        RM_name = 'orbit'
        self.fetch_response_matrix(RM_name=RM_name, orbit=True)
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
