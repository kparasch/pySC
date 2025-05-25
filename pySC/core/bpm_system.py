import numpy as np
from .beam import bpm_reading


class BPMSystem:
    def __init__(self, SC, dual_plane=True):
        if not dual_plane:
            raise NotImplementedError('All BPMs should be dual plane')

        self.SC = SC
        self.indices = SC.ORD.BPM
        first_bpm = list(SC.SIG.BPM.keys())[0]
        sig = SC.SIG.BPM[first_bpm]
        self.rms_errors = {
            'calibration_x' : float(sig.CalError[0]),
            'calibration_y' : float(sig.CalError[1]),
            'offset_x' : float(sig.Offset[0]),
            'offset_y' : float(sig.Offset[1]),
            'roll' : float(sig.Roll),
            'noise_tbt_x' : float(sig.Noise[0]),
            'noise_tbt_y' : float(sig.Noise[1]),
            'noise_co_x' : float(sig.NoiseCO[0]),
            'noise_co_y' : float(sig.NoiseCO[1]),
            }
        
        self.calibration_errors_x = np.array([SC.RING[index].CalError[0] for index in self.indices])
        self.calibration_errors_y = np.array([SC.RING[index].CalError[1] for index in self.indices])
        self.offsets_x = np.array([SC.RING[index].Offset[0] for index in self.indices])
        self.offsets_y = np.array([SC.RING[index].Offset[1] for index in self.indices])
        self.rolls = np.array([SC.RING[index].Roll for index in self.indices])
        

    def capture_orbit(self):
        self.SC.INJ.trackMode = 'ORB'
        orbit, _ = bpm_reading(self.SC)
        return orbit

    def capture_turn_by_turn(self, num_turns=1, return_sigma=False, Z0=None):
        if Z0 is None:
            self.SC.INJ.Z0 = np.zeros(6)
        self.SC.INJ.trackMode = 'TBT'
        SC = self.SC
        SC.INJ.nTurns = num_turns
        delta, sigma = bpm_reading(SC)

        tbt_data = np.full((2, len(SC.ORD.BPM), SC.INJ.nTurns), np.nan)
        for turn in range(self.SC.INJ.nTurns):
            tbt_data[0, :, turn] = delta[0, turn*len(SC.ORD.BPM):(turn+1)*len(SC.ORD.BPM)]

        if return_sigma:
            return tbt_data, sigma
        else:
            return tbt_data
        

