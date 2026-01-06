import numpy as np
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pySC import SimulatedCommissioning

## FACTORIAL[n] = n!
FACTORIAL = np.array([
1,
1,
2,
6,
24,
120,
720,
5040,
40320,
362880,
3628800,
39916800,
479001600,
6227020800,
87178291200,
1307674368000,
20922789888000,
355687428096000,
6402373705728000,
121645100408832000,
2432902008176640000,
])

def binomial_coeff(n:int, k: int):
    return FACTORIAL[n] / (FACTORIAL[k] * FACTORIAL[n-k])

def feeddown(AB: np.ndarray[complex], r0: complex, n: int):
    maxN = len(AB)
    value = 0j
    for k in range(n, maxN):
        value += AB[k] * binomial_coeff(k, n) * (r0 ** (k - n))
    return value 

def get_integrated_strengths_with_feeddown(SC: "SimulatedCommissioning", use_design: bool = False):
    twiss = SC.lattice.get_twiss(use_design=use_design)
    if use_design:
        magnet_settings = SC.design_magnet_settings
    else:
        magnet_settings = SC.magnet_settings

    N = len(twiss['s'])
    temp_max_order = 0
    integrated_strengths = {'norm': {0: np.zeros(N)},
                            'skew': {0: np.zeros(N)}
                           }

    for magnet in magnet_settings.magnets.values():
        ii = magnet.sim_index
        x_co = twiss['x'][ii]
        y_co = twiss['y'][ii]
        if not use_design:
            dx, dy = SC.support_system.get_total_offset(ii)
            roll, _, _ = SC.support_system.get_total_rotation(ii)
        else:
            dx = 0
            dy = 0
            roll = 0

        r0 = dx - x_co + 1.j * (dy - y_co)

        while magnet.max_order > temp_max_order:
            temp_max_order += 1
            integrated_strengths['norm'][temp_max_order] = np.zeros(N)
            integrated_strengths['skew'][temp_max_order] = np.zeros(N)

        AB = (np.array(magnet.B) + 1.j*np.array(magnet.A)) * np.exp(1.j*roll) * magnet.length
        for jj in range(magnet.max_order + 1):
            AB_with_feeddown = feeddown(AB, r0, jj)
            integrated_strengths['norm'][jj][ii] = AB_with_feeddown.real * FACTORIAL[jj]
            integrated_strengths['skew'][jj][ii] = AB_with_feeddown.imag * FACTORIAL[jj]

    return integrated_strengths

def calculate_c_minus(SC: Optional["SimulatedCommissioning"] = None, use_design: bool = False, integrated_strengths : Optional[dict] = None, twiss: Optional[dict] = None):
    if integrated_strengths is None:
        assert SC is not None
        integrated_strengths = get_integrated_strengths_with_feeddown(SC, use_design=use_design)
    ks1l = integrated_strengths['skew'][1]
    if twiss is None:
        assert SC is not None
        twiss = SC.lattice.get_twiss(use_design=use_design)
    integrand = ks1l * np.sqrt(twiss['betx']*twiss['bety']) * np.exp(-1.j*(twiss['mux'] - twiss['muy']))
    c_minus = -np.sum(integrand)/2./np.pi
    return c_minus