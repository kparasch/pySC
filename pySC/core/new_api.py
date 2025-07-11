from .bpm_system import BPMSystem
from .support import SupportSystem

def new_api(SC):

    SC.bpm_system = BPMSystem(SC)
    print(SC.bpm_system.rms_errors)
    
    #orbits_new = np.zeros([100,2,len(bpms.indices)])
    #orbits_old = np.zeros([100,2,len(bpms.indices)])
    #
    # for ii in range(10):
    #     orbits_new[ii] = bpms._capture_orbit()
    #     orbits_old[ii] = bpms.capture_orbit()
    
    #gs, ge = SC.ORD.Girder[:,0]
    
    SC.supports = SupportSystem(parent=SC)
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