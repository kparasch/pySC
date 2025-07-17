from .bpm_system import BPMSystem
from .supports import SupportSystem
import numpy as np

def old_to_new_BPMSystem(SC):
    bpm_system = BPMSystem()
    bpm_system.rng = np.random.default_rng()

    bpm_system.SC = SC
    bpm_system.indices = SC.ORD.BPM
    first_bpm = list(SC.SIG.BPM.keys())[0]
    sig = SC.SIG.BPM[first_bpm]
    bpm_system.rms_errors = {
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
    
    indices = bpm_system.indices
    bpm_system.calibration_errors_x = np.array([SC.RING[index].CalError[0] for index in indices])
    bpm_system.calibration_errors_y = np.array([SC.RING[index].CalError[1] for index in indices])
    bpm_system.offsets_x = np.array([SC.RING[index].Offset[0] for index in indices])
    bpm_system.offsets_y = np.array([SC.RING[index].Offset[1] for index in indices])
    bpm_system.rolls = np.array([SC.RING[index].Roll for index in indices])
    bpm_system.support_offsets_x = np.array([SC.RING[index].SupportOffset[0] for index in indices])
    bpm_system.support_offsets_y = np.array([SC.RING[index].SupportOffset[1] for index in indices])
    bpm_system.support_rolls = np.array([SC.RING[index].SupportRoll for index in indices])

    bpm_system.bba_offsets_x = np.zeros(len(indices))
    bpm_system.bba_offsets_y = np.zeros(len(indices))

    bpm_system.reference_x = np.zeros(len(indices))
    bpm_system.reference_y = np.zeros(len(indices))

    bpm_system.update_offset_and_support()
    return bpm_system

def new_api(SC):

    SC.bpm_system = old_to_new_BPMSystem(SC)
    print(SC.bpm_system.rms_errors)


    SC.supports = SupportSystem()
    SC.supports._parent = SC
    for index in SC.ORD.Magnet:
        SC.supports.add_element(index)
        SC.supports.data['L0'][index].dx = float(SC.RING[index].MagnetOffset[0])
        SC.supports.data['L0'][index].dy = float(SC.RING[index].MagnetOffset[1])
        SC.supports.data['L0'][index].dz = float(SC.RING[index].MagnetOffset[2])
        SC.supports.data['L0'][index].roll = float(SC.RING[index].MagnetRoll[0])
        SC.supports.data['L0'][index].pitch = float(SC.RING[index].MagnetRoll[1])
        SC.supports.data['L0'][index].yaw = float(SC.RING[index].MagnetRoll[2])

    for index in SC.ORD.BPM:
        SC.supports.add_element(index)
        SC.supports.data['L0'][index].dx = float(SC.RING[index].Offset[0])
        SC.supports.data['L0'][index].dy = float(SC.RING[index].Offset[1])
        SC.supports.data['L0'][index].roll = float(SC.RING[index].Roll)

    for ii in range(SC.ORD.Girder.shape[1]):
        SC.supports.add_support(SC.ORD.Girder[0, ii], SC.ORD.Girder[1, ii], name='Girder', level=1)
        sd = SC.RING[SC.ORD.Girder[0,ii]].GirderOffset
        ed = SC.RING[SC.ORD.Girder[1,ii]].GirderOffset
        SC.supports.data['L1'][ii].start.dx = float(sd[0])
        SC.supports.data['L1'][ii].start.dy = float(sd[1])
        SC.supports.data['L1'][ii].end.dx = float(ed[0])
        SC.supports.data['L1'][ii].end.dy = float(ed[1])
        SC.supports.data['L1'][ii].roll = float(SC.RING[SC.ORD.Girder[0,ii]].GirderRoll[0])
    # print(supports)
    SC.supports.resolve_graph()