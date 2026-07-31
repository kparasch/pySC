"""Functions for setting systematic and random multipoles on magnets."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MultipoleTable:
    """Result of reading a multipole table file.

    Attributes:
        AB: array of shape (N, 2), columns are [A, B] components.
            The nominal coefficient (where the original file had ~1.0) is zeroed.
        main_order: 1-based order where the nominal coefficient was found.
        main_component: 'A' or 'B', indicating which column held the nominal.
    """
    AB: np.ndarray
    main_order: int
    main_component: str


def _multipoles_to_dict(multipoles):
    """Convert multipoles to dict form if needed.

    Accepts either a dict of {(component, order): value} or an np.ndarray
    of shape (N, 2) where columns are [A, B].
    """
    if isinstance(multipoles, np.ndarray):
        if multipoles.ndim != 2 or multipoles.shape[1] != 2:
            raise ValueError(f"Expected array of shape (N, 2), got {multipoles.shape}")
        return {
            (comp, row + 1): multipoles[row, col]
            for row in range(multipoles.shape[0])
            for col, comp in enumerate(['A', 'B'])
            if multipoles[row, col] != 0.0
        }
    return multipoles


def _apply_zero_orders(multipoles, zero_orders):
    """Remove entries for specified 1-based orders from multipoles dict."""
    if zero_orders is None:
        return multipoles
    return {
        (comp, order): value
        for (comp, order), value in multipoles.items()
        if order not in zero_orders
    }


def _extend_magnet_lists(magnet, idx):
    """Extend magnet field lists to accommodate index `idx`."""
    if idx >= len(magnet.offset_A):
        new_len = idx + 1
        magnet.offset_A.extend([0.0] * (new_len - len(magnet.offset_A)))
        magnet.offset_B.extend([0.0] * (new_len - len(magnet.offset_B)))
        magnet.A.extend([0.0] * (new_len - len(magnet.A)))
        magnet.B.extend([0.0] * (new_len - len(magnet.B)))
        magnet.max_order = idx


def set_systematic_multipoles(SC, magnet_names, multipoles,
                              relative_to_nominal=True,
                              main_order=None, main_component='B',
                              zero_orders=None):
    """Set systematic multipoles on named magnets.

    Args:
        SC: SimulatedCommissioning instance
        magnet_names: list of magnet names (keys in SC.magnet_settings.magnets)
        multipoles: dict of {(component, order): value} or np.ndarray of shape (N, 2)
            where columns are [A, B]. Order is 1-based (1=dipole, 2=quad, etc.)
        relative_to_nominal: if True, scale value by the nominal main field strength
        main_order: 1-based order of the nominal component.
            Default: magnet.max_order + 1 (i.e. the highest existing order)
        main_component: 'A' or 'B'. Default: 'B'
        zero_orders: list of 1-based orders to exclude before applying
    """
    multipoles = _multipoles_to_dict(multipoles)
    multipoles = _apply_zero_orders(multipoles, zero_orders)

    for name in magnet_names:
        magnet = SC.magnet_settings.magnets[name]

        if relative_to_nominal:
            order_idx = (main_order - 1) if main_order is not None else magnet.max_order
            main_field = abs(SC.lattice.get_magnet_component(
                magnet.sim_index, main_component, order_idx, use_design=True
            ))
            if main_field == 0:
                logger.warning("Magnet '%s' has zero design main field, skipping relative scaling", name)
                main_field = 1.0

        for (component, order), value in multipoles.items():
            idx = order - 1
            _extend_magnet_lists(magnet, idx)

            scaled = value * main_field if relative_to_nominal else value

            if component == 'A':
                magnet.offset_A[idx] += scaled
            elif component == 'B':
                magnet.offset_B[idx] += scaled
            else:
                raise ValueError(f"Component must be 'A' or 'B', got '{component}'")

        magnet.update()
        logger.debug("Set systematic multipoles on '%s'", name)


def set_random_multipoles(SC, magnet_names, multipoles, zero_orders=None):
    """Set random multipoles on named magnets.

    Uses truncated normal distribution (2-sigma) for realistic magnet errors.

    Args:
        SC: SimulatedCommissioning instance
        magnet_names: list of magnet names
        multipoles: dict of {(component, order): rms_value} or np.ndarray of shape (N, 2)
        zero_orders: list of 1-based orders to exclude before applying
    """
    multipoles = _multipoles_to_dict(multipoles)
    multipoles = _apply_zero_orders(multipoles, zero_orders)

    for name in magnet_names:
        magnet = SC.magnet_settings.magnets[name]

        for (component, order), rms_value in multipoles.items():
            idx = order - 1
            _extend_magnet_lists(magnet, idx)

            value = SC.rng.normal_trunc(loc=0, scale=rms_value, sigma_truncate=2)

            if component == 'A':
                magnet.offset_A[idx] += value
            elif component == 'B':
                magnet.offset_B[idx] += value
            else:
                raise ValueError(f"Component must be 'A' or 'B', got '{component}'")

        magnet.update()
        logger.debug("Set random multipoles on '%s'", name)


def read_multipole_table(filepath):
    """Read a multipole table file and return a MultipoleTable.

    Parses tab-or-space-delimited files with format::

        # n    PolynomA(n)    PolynomB(n)
        0      0.0            -0.171
        1      0.0            2.998

    The nominal coefficient (the cell closest to 1.0) is identified and zeroed
    in the returned array to prevent double-counting when used with excitation
    scaling (Phase 2). This matches MATLAB's ``applySysMultipoles`` behavior.

    Args:
        filepath: path to the multipole table file

    Returns:
        MultipoleTable with AB array (nominal zeroed), main_order (1-based),
        and main_component ('A' or 'B')
    """
    filepath = Path(filepath)
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                _order = int(parts[0])
                a_val = float(parts[1])
                b_val = float(parts[2])
                rows.append((a_val, b_val))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No valid data rows found in {filepath}")

    AB = np.array(rows)  # shape (N, 2), cols = [A, B]

    # Find the cell closest to 1.0
    distances = np.abs(AB - 1.0)
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    min_dist = distances[min_idx]

    if min_dist > 1e-6:
        # No cell near 1.0 — fall back to largest magnitude
        mag_idx = np.unravel_index(np.argmax(np.abs(AB)), AB.shape)
        logger.warning(
            "No coefficient close to 1.0 found in %s; defaulting to largest "
            "magnitude at row %d, col %d", filepath, mag_idx[0], mag_idx[1]
        )
        row, col = int(mag_idx[0]), int(mag_idx[1])
    else:
        row, col = int(min_idx[0]), int(min_idx[1])

    main_order = row + 1  # 1-based
    main_component = 'A' if col == 0 else 'B'

    # Zero the nominal coefficient to prevent double-counting
    AB[row, col] = 0.0

    return MultipoleTable(AB=AB, main_order=main_order, main_component=main_component)
