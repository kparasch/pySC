import logging

import numpy as np

logger = logging.getLogger(__name__)


def dynamic_aperture(
    SC, n_angles=16, n_turns=1000, accuracy=1e-6,
    initial_radius=1e-3, max_radius=0.05,
    center_on_orbit=True, use_design=False,
) -> dict:
    """Calculate the dynamic aperture by binary search at each angle."""
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    if center_on_orbit:
        orbit = SC.lattice.get_orbit(use_design=use_design)
        x0, y0 = orbit[0, 0], orbit[1, 0]
    else:
        x0, y0 = 0.0, 0.0

    radii = np.zeros(n_angles)

    for i, theta in enumerate(angles):
        lower = 0.0
        upper = initial_radius

        # Scale upper bound until particle is lost
        while upper <= max_radius:
            if _is_lost(SC, upper, theta, x0, y0, n_turns):
                break
            lower = upper
            upper *= 2.0
        upper = min(upper, max_radius)

        # If even the smallest radius is lost, DA is zero at this angle
        if lower == 0.0 and _is_lost(SC, lower, theta, x0, y0, n_turns):
            radii[i] = 0.0
            continue

        # Binary search
        while upper - lower > accuracy:
            mid = (lower + upper) / 2.0
            if _is_lost(SC, mid, theta, x0, y0, n_turns):
                upper = mid
            else:
                lower = mid

        radii[i] = lower
        logger.debug("Angle %.4f rad: stable radius %.6e m", theta, lower)

    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    area = _shoelace_area(x, y)

    return {"radii": radii, "angles": angles, "area": area, "x": x, "y": y}


def _is_lost(SC, radius, theta, x0, y0, n_turns):
    """Return True if a particle at the given radius and angle is lost."""
    bunch = np.zeros((1, 6))
    bunch[0, 0] = radius * np.cos(theta) + x0
    bunch[0, 2] = radius * np.sin(theta) + y0
    result = SC.lattice.track(bunch, n_turns=n_turns)
    return np.any(np.isnan(result))


def _shoelace_area(x, y):
    """Compute polygon area using the shoelace formula."""
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
