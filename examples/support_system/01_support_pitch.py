"""Inspect an element on a support pitched by moving one endpoint."""

import numpy as np

from pySC import generate_SC
from pySC.core.transformations import at_angles_from_rotation


sc = generate_SC("hmba_config.yaml", seed=1, sigma_truncate=3)

element_name = sc.magnet_arrays["quadrupoles"][0]
element_index = sc.magnet_settings.magnets[element_name].sim_index
family_name = sc.lattice.ring[element_index].FamName
support_index = sc.support_system.add_support(
    element_index - 1,
    element_index + 1,
    level=1,
    name="Test support",
)
sc.support_system.resolve_graph()
sc.support_system.data['L1'][support_index].rigid = True
L = sc.support_system.data['L1'][support_index].length

print(f"Element {family_name} at index {element_index}")
dx, dy, dz = sc.support_system.get_total_offset(element_index)
roll, pitch, yaw = at_angles_from_rotation(sc.support_system.get_total_rotation(element_index))
print(
    f"Before endpoint move: offset [dx, dy, dz] = {np.array([dx, dy, dz])}, "
    f"angles [roll, pitch, yaw] = {np.array([roll, pitch, yaw])}"
)

sc.support_system.set_offset(
    support_index,
    level="L1",
    endpoint="end",
    #dy=L,
    dy=100e-6,
)

dx, dy, dz = sc.support_system.get_total_offset(element_index)
roll, pitch, yaw = at_angles_from_rotation(sc.support_system.get_total_rotation(element_index))
print(
    f"After endpoint move: offset [dx, dy, dz] = {np.array([dx, dy, dz])}, "
    f"angles [roll, pitch, yaw] = {np.array([roll, pitch, yaw])}"
)
