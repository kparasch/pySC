import numpy as np

from pySC.core.SCgetSupportOffset import SCgetSupportOffset


def SCupdateSupport(SC,BPMstructOffset=True,MAGstructOffset=True):
    if MAGstructOffset:
        if 'Magnet' in SC.ORD:
            ords    = np.unique([SC.ORD['Magnet']])
            s       = findspos(SC.RING,ords)
            offsets = SCgetSupportOffset(SC,s)
            rolls   = SCgetSupportRoll(SC,s)
            for i in range(len(ords)):
                ord = ords[i]
                SC.RING[ord].SupportOffset = offsets[:,i]
                SC.RING[ord].SupportRoll = rolls[:,i]
                magLength   = SC.RING[ord].Length
                if 'BendingAngle' in SC.RING[ord]:
                    magTheta = SC.RING[ord].BendingAngle
                else:
                    magTheta = 0
                dx = SC.RING[ord].SupportOffset[0] + SC.RING[ord].MagnetOffset[0]
                dy = SC.RING[ord].SupportOffset[1] + SC.RING[ord].MagnetOffset[1]
                dz = SC.RING[ord].SupportOffset[2] + SC.RING[ord].MagnetOffset[2]
                az = SC.RING[ord].MagnetRoll[0] + SC.RING[ord].SupportRoll[0]
                ax = SC.RING[ord].MagnetRoll[1] + SC.RING[ord].SupportRoll[1]
                ay = SC.RING[ord].MagnetRoll[2] + SC.RING[ord].SupportRoll[2]
                [T1,T2,R1,R2] = SCgetTransformation(dx,dy,dz,ax,ay,az,magTheta,magLength)
                SC.RING[ord].T1 = T1
                SC.RING[ord].T2 = T2
                SC.RING[ord].R1 = R1
                SC.RING[ord].R2 = R2
                if 'MasterOf' in SC.RING[ord]:
                    for childOrd in SC.RING[ord].MasterOf:
                        SC.RING[childOrd].T1=SC.RING[ord].T1
                        SC.RING[childOrd].T2=SC.RING[ord].T2
                        SC.RING[childOrd].R1=SC.RING[ord].R1
                        SC.RING[childOrd].R2=SC.RING[ord].R2
        else:
            print('SC: No magnets have been registered!')
    if BPMstructOffset:
        if 'BPM' in SC.ORD:
            ords    = np.unique([SC.ORD['BPM']])
            s       = findspos(SC.RING,ords)
            offsets = SCgetSupportOffset(SC,s)
            rolls   = SCgetSupportRoll(SC,s)
            for i in range(len(ords)):
                ord = ords[i]
                SC.RING[ord].SupportOffset = offsets[0:2,i] # Longitudinal BPM offsets not yet implemented
                SC.RING[ord].SupportRoll = rolls[0,i] # BPM pitch and yaw angles not yet implemented
        else:
            print('SC: No BPMs have been registered!')
    return SC
# End
