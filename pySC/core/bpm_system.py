from pydantic import BaseModel, PrivateAttr, ConfigDict, model_validator
from typing import TYPE_CHECKING, Optional, Union
from .numpy_type import NPARRAY
import numpy as np

if TYPE_CHECKING:
    from .new_simulated_commissioning import SimulatedCommissioning

BPM_NAME_TYPE = Union[str, int]

def _rotation_matrix(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

BPM_FIELDS_TO_INITIALISE = ['offsets_x', 'offsets_y', 'rolls',
                            'bba_offsets_x', 'bba_offsets_y',
                            'reference_x', 'reference_y']

class BPMSystem(BaseModel):
    indices: list[int] = []
    names: list[BPM_NAME_TYPE] = []

    calibration_errors_x: NPARRAY = np.array([])
    calibration_errors_y: NPARRAY = np.array([])
    offsets_x: NPARRAY = np.array([])
    offsets_y: NPARRAY = np.array([])
    rolls: NPARRAY = np.array([])

    noise_co_x: NPARRAY = np.array([])
    noise_co_y: NPARRAY = np.array([])
    noise_tbt_x: NPARRAY = np.array([])
    noise_tbt_y: NPARRAY = np.array([])

    bba_offsets_x: NPARRAY = np.array([])
    bba_offsets_y: NPARRAY = np.array([])
    reference_x: NPARRAY = np.array([])
    reference_y: NPARRAY = np.array([])

    _parent: Optional["SimulatedCommissioning"] = PrivateAttr(default=None)
    _rot_matrices: Optional[NPARRAY] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)


    @model_validator(mode="after")
    def initialize(self):
        if len(self.rolls) > 0:
            self.update_rot_matices()
        return self

    def update_rot_matrices(self):
        self._rot_matrices = _rotation_matrix(self.rolls)

    def bpm_number(self, index):
        return int(np.where(np.array(self.indices) == index)[0][0])

    def capture_orbit(self, bba=True, subtract_reference=True) -> tuple[np.ndarray, np.ndarray]:
        '''
        Simulates an orbit reading from the BPMs, applying calibration errors, offsets/rolls, and noise.
        Args:
            bba (bool): If True, corrects for the BBA offsets stored in the BPMSystem class.
            subtract_reference (bool): If True, subtracts the reference orbit from the simulated orbit.
        Returns:
            fake_orbit_x: Simulated x-coordinates of the orbit at the BPMs.
            fake_orbit_y: Simulated y-coordinates of the orbit at the BPMs.
        '''
        orbit = self._parent.lattice.get_orbit(indices=self.indices)
        rotated_orbit = np.einsum('ijk,jk->ik', self._rot_matrices, orbit)  # Rotate orbit according to bpm roll

        noise_x = self._parent.rng.normal(scale=self.noise_co_x)
        noise_y = self._parent.rng.normal(scale=self.noise_co_y)

        fake_orbit_x = (rotated_orbit[0] - self.offsets_x) * (1 + self.calibration_errors_x) + noise_x
        fake_orbit_y = (rotated_orbit[1] - self.offsets_y) * (1 + self.calibration_errors_y) + noise_y

        if bba:
            # Apply BBA offsets
            fake_orbit_x -= self.bba_offsets_x
            fake_orbit_y -= self.bba_offsets_y

        if subtract_reference:
            # Subtract reference orbit
            fake_orbit_x -= self.reference_x
            fake_orbit_y -= self.reference_y

        return fake_orbit_x, fake_orbit_y

    def capture_injection(self, n_turns=1, bba=True, subtract_reference=True) -> tuple[np.ndarray, np.ndarray]:
        '''
        Simulates an orbit reading during injection from the BPMs, applying calibration errors, offsets/rolls, and noise.
        Args:
            bba (bool): If True, corrects for the BBA offsets stored in the BPMSystem class.
            subtract_reference (bool): If True, subtracts the reference orbit from the simulated orbit.
        Returns:
            fake_orbit_x: Simulated x-coordinates of the orbit at the BPMs.
            fake_orbit_y: Simulated y-coordinates of the orbit at the BPMs.
        '''
        #orbit = at_wrapper.get_orbit(self.SC.RING, self.indices)
        bunch = self._parent.injection.generate_bunch()
        trajectory = np.mean(self._parent.lattice.track(bunch, indices=self.indices, use_design=False), axis=1) # average over all particles

        fake_trajectory_x_tbt = np.zeros([len(self.indices), n_turns])
        fake_trajectory_y_tbt = np.zeros([len(self.indices), n_turns])

        for n in range(n_turns):
            one_trajectory = trajectory[:, :, n]
            rotated_trajectory = np.einsum('ijk,jk->ik', self._rot_matrices, one_trajectory)  

            noise_x = self._parent.rng.normal(scale=self.noise_tbt_x)
            noise_y = self._parent.rng.normal(scale=self.noise_tbt_y)

            fake_trajectory_x = (rotated_trajectory[0] - self.offsets_x) * (1 + self.calibration_errors_x) + noise_x
            fake_trajectory_y = (rotated_trajectory[1] - self.offsets_y) * (1 + self.calibration_errors_y) + noise_y

            if bba:
                # Apply BBA offsets
                fake_trajectory_x -= self.bba_offsets_x
                fake_trajectory_y -= self.bba_offsets_y

            if subtract_reference:
                # Subtract reference orbit
                fake_trajectory_x -= self.reference_x
                fake_trajectory_y -= self.reference_y

            fake_trajectory_x_tbt[:, n] = fake_trajectory_x
            fake_trajectory_y_tbt[:, n] = fake_trajectory_y

        return fake_trajectory_x_tbt, fake_trajectory_y_tbt

    # def capture_turn_by_turn(self, num_turns=1, return_sigma=False, Z0=None):
    #     if Z0 is None:
    #         self.SC.INJ.Z0 = np.zeros(6)
    #     self.SC.INJ.trackMode = 'TBT'
    #     SC = self.SC
    #     SC.INJ.nTurns = num_turns
    #     delta, sigma = bpm_reading(SC)

    #     tbt_data = np.full((2, len(SC.ORD.BPM), SC.INJ.nTurns), np.nan)
    #     for turn in range(self.SC.INJ.nTurns):
    #         tbt_data[0, :, turn] = delta[0, turn*len(SC.ORD.BPM):(turn+1)*len(SC.ORD.BPM)]

    #     if return_sigma:
    #         return tbt_data, sigma
    #     else:
    #         return tbt_data


