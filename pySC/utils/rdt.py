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

S4 = np.array([[0., 1., 0., 0.],
              [-1., 0., 0., 0.],
              [ 0., 0., 0., 1.],
              [ 0., 0.,-1., 0.]])

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

def calculate_c_minus(SC: Optional["SimulatedCommissioning"] = None, use_design: bool = False):

    M = SC.lattice.one_turn_matrix(use_design=use_design)
    W, _, _, q1, q2 = linear_normal_form(M)

    c_r1 = np.sqrt(W[2,0]**2 + W[2,1]**2) / W[0,0]
    c_r2 = np.sqrt(W[0,2]**2 + W[0,3]**2) / W[2,2]
    c_phi1 = np.arctan2(W[2,1], W[2,0])

    cmin_amp = (2 * np.sqrt(c_r1*c_r2) * np.abs(q1 - q2) / (1 + c_r1 * c_r2))
    c_minus = cmin_amp * np.exp(1j * c_phi1)

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


def Rot2D(mu):
    return np.array([[ np.cos(mu), np.sin(mu)],
                     [-np.sin(mu), np.cos(mu)]])

def linear_normal_form(M):
    w0, v0 = np.linalg.eig(M[:4,:4])

    a0 = np.real(v0)
    b0 = np.imag(v0)

    index_list = [0,1,2,3]

    ##### Sort modes in pairs of conjugate modes #####

    conj_modes = np.zeros([2,2], dtype=int)

    conj_modes[0,0] = index_list[0]
    del index_list[0]

    min_index = 0 
    min_diff = abs(np.imag(w0[conj_modes[0,0]] + w0[index_list[min_index]]))
    for i in range(1,len(index_list)):
        diff = abs(np.imag(w0[conj_modes[0,0]] + w0[index_list[i]]))
        if min_diff > diff:
            min_diff = diff
            min_index = i

    conj_modes[0,1] = index_list[min_index]
    del index_list[min_index]

    conj_modes[1,0] = index_list[0]
    conj_modes[1,1] = index_list[1]

    ##################################################
    #### Select mode from pairs with positive (real @ S @ imag) #####

    modes = np.empty(2, dtype=int)
    for ii,ind in enumerate(conj_modes):
        if np.matmul(np.matmul(a0[:,ind[0]], S4), b0[:,ind[0]]) > 0:
            modes[ii] = ind[0]
        else:
            modes[ii] = ind[1]

    ##################################################
    #### Sort modes such that (1,2) is close to (x,y) ####

    if abs(v0[:,modes[1]])[2] < abs(v0[:,modes[0]])[2]:
        modes[0], modes[1] = modes[1], modes[0]

    ##################################################
    #### Rotate eigenvectors to the Courant-Snyder parameterization ####
    phase0 = np.log(v0[0,modes[0]]).imag
    phase1 = np.log(v0[2,modes[1]]).imag

    v0[:,modes[0]] *= np.exp(-1.j*phase0)
    v0[:,modes[1]] *= np.exp(-1.j*phase1)

    ##################################################
    #### Construct W #################################

    a1 = v0[:,modes[0]].real
    a2 = v0[:,modes[1]].real
    b1 = v0[:,modes[0]].imag
    b2 = v0[:,modes[1]].imag

    n1 = 1./np.sqrt(np.matmul(np.matmul(a1, S4), b1))
    n2 = 1./np.sqrt(np.matmul(np.matmul(a2, S4), b2))

    a1 *= n1
    a2 *= n2

    b1 *= n1
    b2 *= n2

    W = np.array([a1,b1,a2,b2]).T
    W[abs(W) < 1.e-14] = 0. # Set very small numbers to zero.
    invW = np.matmul(np.matmul(S4.T, W.T), S4)

    ##################################################
    #### Get tunes and rotation matrix in the normalized coordinates ####

    mu1 = np.log(w0[modes[0]]).imag
    mu2 = np.log(w0[modes[1]]).imag

    q1 = mu1/(2.*np.pi)
    q2 = mu2/(2.*np.pi)

    R = np.zeros_like(W)
    R[0:2,0:2] = Rot2D(mu1)
    R[2:4,2:4] = Rot2D(mu2)
    ##################################################    

    return W, invW, R, q1, q2