import numpy as np

from pySC.utils import logging_tools
from pySC.core.beam import bpm_reading

LOGGER = logging_tools.get_logger(__name__)

def modelDispersion(SC):

    _, _, twiss = SC.IDEALRING.get_optics(refpts=SC.ORD.BPM)
    Dx = twiss.dispersion[:, 0]  # horizontal dispersion
    Dy = twiss.dispersion[:, 2]  # vertical dispersion
    return Dx, Dy

def measureDispersion(SC, CAVords, rfStep=50):
    """
    Calculates the lattice dispersion (orbit-based) based on current setpoints

    Calculates the dispersion at the bpms by changing the frequency of the rf cavities
    specified in `CAVords` using the current magnet setpoints.

    Args:
        SC:
            SimulatedCommissioning class instance
        CAVords:
            Index of (main) RF Cavities in SC.RING (SC.ORD.CM)
        rfStep: (default = 50) Change of rf frequency [Hz]

    Returns:
        Dx : The horizontal dispersion given in [m].
        Dy : The vertical dispersion given in [m].

    """
    LOGGER.info('Calculating model dispersion')

    assert SC.INJ.trackMode == 'ORB', "Track mode must be 'ORB' for dispersion calculation"

    T0, _ = bpm_reading(SC)

    if np.any(np.isnan(T0)):
        raise ValueError('Initial trajectory/orbit is NaN. Aborting. ')

    for ord in CAVords:  # Single point with all cavities with the same frequency shift
        SC.RING[ord].Frequency += rfStep

    T1, _ = bpm_reading(SC)

    for ord in CAVords:  # Single point with all cavities with the same frequency shift
        SC.RING[ord].Frequency -= rfStep

    if np.any(np.isnan(T1)):
        raise ValueError('Final trajectory/orbit is NaN. Aborting. ')

    f_rf = np.mean([SC.RING[ord].Frequency for ord in CAVords])  # Average frequency of all cavities

    ## get momentum compaction factor
    SC.IDEALRING.disable_6d()
    momentum_compaction_factor = SC.IDEALRING.get_mcf()
    SC.IDEALRING.enable_6d()
    ### 
    gamma0 = SC.IDEALRING.gamma
    momentum_deviation_difference = - rfStep / f_rf / (momentum_compaction_factor - 1 / gamma0**2)
    LOGGER.info(f"Momentum deviation difference: {momentum_deviation_difference:.6f}")

    Dx = (T1[0] - T0[0]) / momentum_deviation_difference 
    Dy = (T1[1] - T0[1]) / momentum_deviation_difference

    return Dx, Dy