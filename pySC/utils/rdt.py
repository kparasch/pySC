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

def omega(x):
    if x%2:
        return 0
    else:
        return 1

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
    Delta = twiss['qx'] - twiss['qy']
    integrand = ks1l * np.sqrt(twiss['betx']*twiss['bety']) * np.exp(+1.j*(twiss['mux'] - twiss['muy'] - np.pi*Delta))
    #integrand = ks1l * np.sqrt(twiss['betx']*twiss['bety']) * np.exp(-1.j*(twiss['mux'] - twiss['muy'] - 2*np.pi*Delta*twiss['s']/circumference))
    c_minus = np.sum(integrand)/2./np.pi
    return c_minus

def hjklm(SC: Optional["SimulatedCommissioning"] = None, j: int = 0, k: int = 0, l: int = 0, m: int = 0, use_design: bool = False,
          integrated_strengths: Optional[dict] = None, twiss: Optional[dict] = None):
    n = j+k+l+m
    assert n > 0

    if integrated_strengths is None:
        assert SC is not None
        integrated_strengths = get_integrated_strengths_with_feeddown(SC, use_design=use_design)
    if twiss is None:
        assert SC is not None
        twiss = SC.lattice.get_twiss(use_design=use_design)

    K = integrated_strengths['norm'][n-1]
    J = integrated_strengths['skew'][n-1]
    h = - (K * omega(l+m) + 1j * J * omega(l+m+1))/(FACTORIAL[j] * FACTORIAL[k] * FACTORIAL[l] * FACTORIAL[m] * 2**(n)) * (1.j)**(l+m) * twiss['betx']**((j+k)/2) * twiss['bety']**((l+m)/2)
    return h


def fjklm(SC: Optional["SimulatedCommissioning"] = None, j: int = 0, k: int = 0, l: int = 0, m: int = 0,
          use_design: bool = False, integrated_strengths: Optional[dict] = None, twiss: Optional[dict] = None, normalized: bool = True):

    assert j + k + l + m > 0

    if twiss is None:
        assert SC is not None
        twiss = SC.lattice.get_twiss(use_design=use_design)

    qx = twiss['qx']
    qy = twiss['qy']
    denom = 1 - np.exp(1.j * 2 * np.pi * ((j-k) * qx + (l-m) * qy))
    h = hjklm(SC=SC, j=j, k=k, l=l, m=m, use_design=use_design, integrated_strengths=integrated_strengths, twiss=twiss)
    mask = h != 0
    hm = h[mask]
    mux = twiss['mux'][mask]
    muy = twiss['muy'][mask]
    ii = 0
    f = np.zeros_like(twiss['s'], dtype=complex)
    for ii in range(len(twiss['s'])):
        dphix = 2*np.pi*np.abs(twiss['mux'][ii] - mux)
        dphiy = 2*np.pi*np.abs(twiss['muy'][ii] - muy)
        expo = np.exp(1.j * ( (j-k) * dphix + (l-m) * dphiy))
        f[ii] = np.sum(hm * expo)
    if normalized:
        return f / denom
    else:
        return f