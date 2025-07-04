import copy
import numpy as np
from at import Lattice

from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.core.constants import NUM_TO_AB, RF_PROPERTIES
from pySC.utils import at_wrapper, logging_tools

LOGGER = logging_tools.get_logger(__name__)

def SCgetModelRM(SC, BPMords, CMords, trackMode='TBT', Z0=np.zeros(6), nTurns=1, dkick=1e-5, useIdealRing=True):
    """
    Determine the lattice response matrix based on current setpoints.

    SCgetModelRM calculates the response matrix `RM` with the BPMs at the ordinates `BPMords`
    and corrector magnets at ordinates `CMords` using the current magnet setpoints without any
    roll/alignment/calibration errors. `CMords` is a 2-cell array containing the two lists for the
    horizontal and vertical CMs respectively.
    This routine can determine the turn-by-turn RM, as well as the orbit-RM; see option 'trackMode'.

    Args:
        SC:
            SimulatedCommissioning class instance
        BPMords:
            Index of BPMs in SC.RING (SC.ORD.BPM)
        CMords:
            Index of Correctors Magnets in SC.RING (SC.ORD.CM)
        trackMode:
            (default = 'TBT') If `TBT` the turn-by-turn RM is calculated.
            If `ORB` the orbit RM is calculated, using `at.findorbit6`
        Z0:
            (default = numpy.zeros(6)) Initial condition for tracking.
            In `ORB`-mode this is used as the initial guess for `findorbit6`.
        nTurns:
            (default = 1) Number of turns over which to determine the TBT-RM. Ignored if in `ORB`-mode.
        dkick:
            (default = 1e-5) Kick (scalar or array-like inputs) [rad] to be added when numerically determining the partial derivatives.
        useIdealRing:
            (default = True) If True, the design lattice specified in `SC.IDEALRING` is used.
            If False, the model lattice is used SCgetModelRING(SC).

    Returns:
        The response matrix given in [m/rad].

    Examples:
        Compute a response matrix::

            RM1 = SCgetModelRM(SC, SC.ORD.BPM, SC.ORD.CM, nTurns=1)

    """
    LOGGER.info('Calculating model response matrix')
    track_methods = dict(TBT=at_wrapper.lattice_track, ORB=orbpass)
    if trackMode not in track_methods.keys():
        ValueError(f'Unknown track mode {trackMode}. Valid values are {track_methods.keys()}')
    ring = SC.IDEALRING.deepcopy() if useIdealRing else SCgetModelRING(SC)
    trackmethod = track_methods[trackMode]
    if trackMode == 'ORB':
        nTurns = 1
    nBPM = len(BPMords)
    nCM = len(CMords[0]) + len(CMords[1])
    RM = np.full((2 * nBPM * nTurns, nCM), np.nan)
    Ta = trackmethod(ring, Z0,  nTurns, BPMords)
    if np.any(np.isnan(Ta)):
        raise ValueError('Initial trajectory/orbit is NaN. Aborting. ')
    cnt = 0
    for nDim in range(2):  # 0: Horizontal, 1: Vertical
        for j, CMord in enumerate(CMords[nDim]):
            this_dkick = dkick[nDim][j] if isinstance(dkick, (list, tuple, np.ndarray)) else dkick
            if ring[CMord].PassMethod == 'CorrectorPass':
                KickNominal = ring[CMord].KickAngle[nDim]
                ring[CMord].KickAngle[nDim] = KickNominal + this_dkick
                TdB = trackmethod(ring, Z0, nTurns, BPMords)
                ring[CMord].KickAngle[nDim] = KickNominal
            else:
                PolynomNominal = getattr(ring[CMord], f"Polynom{NUM_TO_AB[nDim]}")
                delta = this_dkick / ring[CMord].Length
                changed_polynom = copy.deepcopy(PolynomNominal[:])
                changed_polynom[0] += (-1) ** (nDim + 1) * delta
                setattr(ring[CMord], f"Polynom{NUM_TO_AB[nDim]}", changed_polynom[:])
                TdB = trackmethod(ring, Z0, nTurns, BPMords)
                setattr(ring[CMord], f"Polynom{NUM_TO_AB[nDim]}", PolynomNominal[:])
            dTdB = (TdB - Ta) / this_dkick
            RM[:, cnt] = np.concatenate((np.ravel(np.transpose(dTdB[0, :, :, :], axes=(2, 1, 0))),
                                         np.ravel(np.transpose(dTdB[2, :, :, :], axes=(2, 1, 0)))))
            cnt += 1
    return RM


def orbpass(RING, Z0,  nTurns, REFPTS):
    return np.transpose(at_wrapper.findorbit6(RING, REFPTS)[1])[[0,1,2,3], :].reshape(4, 1, len(REFPTS), 1)


def SCgetModelDispersion(SC, BPMords, CAVords, trackMode='ORB', Z0=np.zeros(6), nTurns=1, rfStep=1E3, useIdealRing=True):
    """
    Calculates the lattice dispersion based on current setpoints

    Calculates the dispersion at the ordinates `BPMords` by changing the frequency of the rf cavities
    specified in `CAVords` using the current magnet setpoints without any roll/alignment/calibration
    errors. Optionally the design lattice is used.

    Args:
        SC:
            SimulatedCommissioning class instance
        BPMords:
            Index of BPMs in SC.RING (SC.ORD.BPM)
        CAVords:
            Index of RF Cavities in SC.RING (SC.ORD.CM)
        trackMode:
            (default = 'TBT') If `TBT` the turn-by-turn RM is calculated.
            If `ORB` the orbit RM is calculated, using `at.findorbit6`
        Z0:
            (default = numpy.zeros(6)) Initial condition for tracking.
            In `ORB`-mode this is used as the initial guess for `findorbit6`.
        nTurns:
            (default = 1) Number of turns over which to determine the TBT-RM. Ignored if in `ORB`-mode.
        rfStep: (default = 1e3) Change of rf frequency [Hz]
        useIdealRing: (default=False) If true, the design lattice specified in `SC.IDEALRING` is used.

    Returns:
        The dispersion given in [m/Hz].

    """
    LOGGER.info('Calculating model dispersion')
    track_methods = dict(TBT=at_wrapper.lattice_track, ORB=orbpass)
    if trackMode not in track_methods.keys():
        ValueError(f'Unknown track mode {trackMode}. Valid values are {track_methods.keys()}')
    ring = SC.IDEALRING.deepcopy() if useIdealRing else SCgetModelRING(SC)
    trackmethod = track_methods[trackMode]
    if trackMode == 'ORB':
        nTurns = 1

    Ta = trackmethod(ring, Z0,  nTurns, BPMords)
    if np.any(np.isnan(Ta)):
        raise ValueError('Initial trajectory/orbit is NaN. Aborting. ')

    for ord in CAVords:  # Single point with all cavities with the same frequency shift
        ring[ord].Frequency += rfStep
    TdB = trackmethod(ring, Z0,  nTurns, BPMords)
    dTdB = (TdB - Ta) / rfStep
    eta = np.concatenate((np.ravel(np.transpose(dTdB[0, :, :, :], axes=(2, 1, 0))),
                          np.ravel(np.transpose(dTdB[2, :, :, :], axes=(2, 1, 0)))))
    return eta


def SCgetModelRING(SC: SimulatedCommissioning, includeAperture: bool =False) -> Lattice:
    """
    Returns a model lattice based on current setpoints

    This function calculates a model lattice based on the setpoints of `SC.RING`. Misalignments,
    lattice errors and dipole fields are excluded.

    Args:
        SC: SimulatedCommissioning class instance
        includeAperture: (default=False) If true, the returned model ring includes the aperture

    Returns:
        The idealised RING structure

    """
    ring = SC.IDEALRING.deepcopy()
    for ord in range(len(SC.RING)):
        if hasattr(SC.RING[ord], 'SetPointA') and hasattr(SC.RING[ord], 'SetPointB'):
            ring[ord].PolynomA = SC.RING[ord].SetPointA
            ring[ord].PolynomB = SC.RING[ord].SetPointB
            ring[ord].PolynomA[0] = 0.0
            ring[ord].PolynomB[0] = 0.0
        if includeAperture:
            if 'EApertures' in SC.RING[ord]:
                ring[ord].EApertures = SC.RING[ord].EApertures
            if 'RApertures' in SC.RING[ord]:
                ring[ord].RApertures = SC.RING[ord].RApertures
        if len(SC.ORD.RF) and hasattr(SC.RING[ord], 'Frequency'):
            for field in RF_PROPERTIES:
                setattr(ring[ord], field, getattr(SC.RING[ord], f"{field}SetPoint"))
    return ring
