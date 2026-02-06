from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..core.new_simulated_commissioning import SimulatedCommissioning


def get_average_orbit(SC: "SimulatedCommissioning", n_shots: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Measure and return the average orbit and noise over a specified number of shots.

    Parameters
    ----------
    SC : SimulatedCommissioning
        The SimulatedCommissioning instance.
    n_shots : int, optional
        The number of shots to average over, by default 10.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - x_avg: Average x positions at BPMs.
        - y_avg: Average y positions at BPMs.
        - x_std: Standard deviation of x positions at BPMs (noise).
        - y_std: Standard deviation of y positions at BPMs (noise).
    """
    all_orbit_x = np.zeros((n_shots, len(SC.bpm_system.names)))
    all_orbit_y = np.zeros((n_shots, len(SC.bpm_system.names)))

    for i in range(n_shots):
        x, y = SC.bpm_system.capture_orbit(use_design=False)
        all_orbit_x[i, :] = x
        all_orbit_y[i, :] = y

    x_avg = np.nanmean(all_orbit_x, axis=0)
    y_avg = np.nanmean(all_orbit_y, axis=0)

    x_std = np.nanstd(all_orbit_x, axis=0)
    y_std = np.nanstd(all_orbit_y, axis=0)
    return x_avg, y_avg, x_std, y_std