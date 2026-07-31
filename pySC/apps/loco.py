"""
LOCO (Linear Optics from Closed Orbits) implementation for pySC.

Provides functions to compute orbit response matrix Jacobians with respect
to quadrupole strengths, fit optics corrections via least-squares, and
apply the resulting corrections back to the lattice.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

from ..tuning.response_measurements import (
    measure_OrbitResponseMatrix,
    measure_RFFrequencyOrbitResponse,
)

if TYPE_CHECKING:
    from ..core.simulated_commissioning import SimulatedCommissioning

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _objective(masked_params, orm_residuals, masked_jacobian, weights):
    """Weighted residual for the LOCO least-squares problem."""
    residuals = orm_residuals - np.einsum("ijk,i->jk", masked_jacobian, masked_params)
    w = np.sqrt(np.atleast_1d(weights))
    return (residuals * w[:, np.newaxis]).ravel()


def _get_parameters_mask(fit_flags, lengths):
    """Build a boolean mask selecting the parameter groups to fit."""
    mask = np.zeros(sum(lengths), dtype=bool)
    current_index = 0
    for flag, length in zip(fit_flags, lengths):
        if flag:
            mask[current_index : current_index + length] = True
        current_index += length
    return mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_jacobian(
    SC: "SimulatedCommissioning",
    orm_model,
    quad_control_names,
    dk=1e-5,
    include_correctors=True,
    include_bpms=True,
    include_dispersion=False,
    use_design=True,
):
    """Compute the LOCO Jacobian of the ORM with respect to parameters.

    Parameters
    ----------
    SC : SimulatedCommissioning
        The simulation object.
    orm_model : np.ndarray
        Model orbit response matrix, shape ``(n_bpms*2, n_correctors)``.
        If *include_dispersion* is True the last column should be the
        dispersion response.
    quad_control_names : list[str]
        Names of the quadrupole control knobs to include.
    dk : float
        Finite-difference step size for quadrupole strengths.
    include_correctors : bool
        Whether to append corrector-gain Jacobian entries.
    include_bpms : bool
        Whether to append BPM-gain Jacobian entries.
    include_dispersion : bool
        Whether to append the RF-frequency orbit response (dispersion)
        as an extra column in each ORM measurement.
    use_design : bool
        If True, compute model ORMs from the design lattice.

    Returns
    -------
    jacobian : np.ndarray
        Shape ``(n_params, n_bpms*2, n_correctors [+1])``.
    """
    n_rows, n_cols = orm_model.shape
    jacobian_rows = []

    # --- Quadrupole derivatives ---
    for name in quad_control_names:
        current = SC.magnet_settings.get(name)
        SC.magnet_settings.set(name, current + dk)

        orm_bumped = measure_OrbitResponseMatrix(SC, use_design=use_design)

        if include_dispersion:
            disp = measure_RFFrequencyOrbitResponse(SC, use_design=use_design)
            disp_col = disp.reshape(-1, 1)
            orm_bumped = np.hstack([orm_bumped, disp_col])

        SC.magnet_settings.set(name, current)

        jacobian_rows.append((orm_bumped - orm_model) / dk)

    # --- Corrector-gain entries ---
    if include_correctors:
        for j in range(n_cols):
            entry = np.zeros_like(orm_model)
            entry[:, j] = orm_model[:, j]
            jacobian_rows.append(entry)

    # --- BPM-gain entries ---
    if include_bpms:
        for i in range(n_rows):
            entry = np.zeros_like(orm_model)
            entry[i, :] = orm_model[i, :]
            jacobian_rows.append(entry)

    jacobian = np.array(jacobian_rows)
    logger.info(
        "LOCO Jacobian computed: %d parameters, ORM shape (%d, %d)",
        jacobian.shape[0],
        n_rows,
        n_cols,
    )
    return jacobian


def loco_fit(
    SC: "SimulatedCommissioning",
    orm_measured,
    orm_model,
    jacobian,
    fit_quads=True,
    fit_correctors=True,
    fit_bpms=True,
    s_cut=None,
    weights=1,
    method="lm",
):
    """Fit LOCO corrections using least-squares.

    Parameters
    ----------
    SC : SimulatedCommissioning
        The simulation object (used only to infer dimension info).
    orm_measured : np.ndarray
        Measured orbit response matrix.
    orm_model : np.ndarray
        Model orbit response matrix.
    jacobian : np.ndarray
        Jacobian from :func:`calculate_jacobian`.
    fit_quads, fit_correctors, fit_bpms : bool
        Which parameter groups to fit.
    s_cut : float or None
        Singular-value cut-off (not used by ``method='lm'``; reserved for
        future SVD-based solvers).
    weights : float or np.ndarray
        Weights applied to the residual.
    method : str
        Optimisation method forwarded to :func:`scipy.optimize.least_squares`.

    Returns
    -------
    dict
        ``'quad_corrections'``, ``'corrector_corrections'``,
        ``'bpm_corrections'`` — each a numpy array (zeros where the
        corresponding group was not fitted).
    """
    n_rows, n_cols = orm_model.shape
    n_total = jacobian.shape[0]

    # Jacobian layout: [quads, correctors, bpms]
    # Corrector and BPM blocks are always present in the Jacobian
    n_quad = n_total - n_cols - n_rows
    lengths = [n_quad, n_cols, n_rows]
    mask = _get_parameters_mask([fit_quads, fit_correctors, fit_bpms], lengths)

    initial_guess = np.zeros(n_total)

    result = least_squares(
        lambda delta_params: _objective(
            delta_params, orm_measured - orm_model, jacobian[mask], weights
        ),
        initial_guess[mask],
        method=method,
        verbose=0,
    )
    corrections = result.x

    # Unpack into groups
    quad_corrections = np.zeros(n_quad)
    corrector_corrections = np.zeros(n_cols)
    bpm_corrections = np.zeros(n_rows)

    idx = 0
    if fit_quads:
        quad_corrections[:] = corrections[idx : idx + n_quad]
        idx += n_quad
    if fit_correctors:
        corrector_corrections[:] = corrections[idx : idx + n_cols]
        idx += n_cols
    if fit_bpms:
        bpm_corrections[:] = corrections[idx : idx + n_rows]
        idx += n_rows

    logger.info(
        "LOCO fit complete (method=%s): cost=%.6e, nfev=%d",
        method,
        result.cost,
        result.nfev,
    )

    return {
        "quad_corrections": quad_corrections,
        "corrector_corrections": corrector_corrections,
        "bpm_corrections": bpm_corrections,
    }


def apply_loco_corrections(SC, corrections, quad_control_names, fraction=1.0):
    """Apply fitted LOCO corrections to the lattice.

    Parameters
    ----------
    SC : SimulatedCommissioning
        The simulation object.
    corrections : dict
        Output of :func:`loco_fit`.
    quad_control_names : list[str]
        Same list used in :func:`calculate_jacobian`.
    fraction : float
        Fraction of the correction to apply (useful for iterative schemes).
    """
    for name, dk in zip(quad_control_names, corrections["quad_corrections"]):
        current = SC.magnet_settings.get(name)
        SC.magnet_settings.set(name, current - fraction * dk)

    logger.info(
        "Applied LOCO corrections (fraction=%.2f) to %d quads",
        fraction,
        len(quad_control_names),
    )


def measure_orm(SC, dkick=1e-5, use_design=False):
    """Convenience wrapper around :func:`measure_OrbitResponseMatrix`.

    Parameters
    ----------
    SC : SimulatedCommissioning
        The simulation object.
    dkick : float
        Kick size for the finite-difference ORM measurement.
    use_design : bool
        If True, use the design lattice.

    Returns
    -------
    np.ndarray
        Orbit response matrix.
    """
    return measure_OrbitResponseMatrix(SC, dkick=dkick, use_design=use_design)


def analyze_ring(SC, use_design=False):
    """Compute summary optics figures of merit.

    Parameters
    ----------
    SC : SimulatedCommissioning
        The simulation object.
    use_design : bool
        If True, analyse the design lattice (useful as a reference baseline
        — in that case beta-beat and dispersion errors will be zero).

    Returns
    -------
    dict
        Keys: ``orbit_rms_x``, ``orbit_rms_y``, ``beta_beat_x``,
        ``beta_beat_y``, ``disp_err_x``, ``disp_err_y``, ``tune_x``,
        ``tune_y``, ``chromaticity_x``, ``chromaticity_y``.
    """
    twiss = SC.lattice.get_twiss(use_design=use_design)
    design = SC.lattice.get_twiss(use_design=True)
    orbit = SC.lattice.get_orbit(use_design=use_design)

    return {
        "orbit_rms_x": np.std(orbit[0]),
        "orbit_rms_y": np.std(orbit[1]),
        "beta_beat_x": np.std((twiss["betx"] - design["betx"]) / design["betx"]),
        "beta_beat_y": np.std((twiss["bety"] - design["bety"]) / design["bety"]),
        "disp_err_x": np.std(twiss["dx"] - design["dx"]),
        "disp_err_y": np.std(twiss["dy"] - design["dy"]),
        "tune_x": twiss["qx"],
        "tune_y": twiss["qy"],
        "chromaticity_x": twiss["dqx"],
        "chromaticity_y": twiss["dqy"],
    }
