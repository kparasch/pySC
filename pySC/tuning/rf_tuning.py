from typing import Optional, TYPE_CHECKING
from pydantic import BaseModel, PrivateAttr
import numpy as np
import logging

if TYPE_CHECKING:
    from .tuning_core import Tuning

logger = logging.getLogger(__name__)

class RF_tuning(BaseModel, extra="forbid"):
    phase_pickup_index: int = 0
    _parent: Optional['Tuning'] = PrivateAttr(default=None)

    def measure_injection_phase(self, n_turns: int = 1) -> float:
        SC = self._parent._parent
        bunch = SC.injection.generate_bunch()
        x, tau = SC.lattice.track(bunch, indices=[self.phase_pickup_index], n_turns=n_turns, coordinates=['x', 'tau'])
        mask = x != x
        tau[mask] = np.nan

        mean_tau = np.nanmean(tau, axis=0)
        return mean_tau

    def optimize_phase(self, low: int = -180, high: int = 180, npoints: int = 10, n_turns: int = 10) -> float:
        logger.info(f"Optimizing RF phase in range [{low}, {high}] deg. with {npoints} points and {n_turns} turns")
        SC = self._parent._parent
        best_phase = SC.rf_settings.main.phase
        best = np.inf
        for phase in np.linspace(low, high, npoints):
            SC.rf_settings.main.set_phase(phase)
            out = np.nanstd(SC.tuning.rf.measure_injection_phase(n_turns=n_turns)*1e6)
            if out < best:
                best_phase = phase
                best = out
            logger.info(f"Phase {phase:.1f} deg., r.m.s.(tau) =  {out:.1f} um")
        logger.info(f"Setting to optimal phase: {best_phase:.1f} deg.")
        SC.rf_settings.main.set_phase(best_phase)
        return best_phase