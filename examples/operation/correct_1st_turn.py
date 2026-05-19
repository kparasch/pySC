"""Correct the first-turn trajectory using an operational interface."""

import json

import numpy as np

from interface import InterfaceInjection
from pySC import ResponseMatrix, orbit_correction


N_TURNS = 1
APPLY_CORRECTION = True

interface = InterfaceInjection(n_turns=N_TURNS)

response_matrix = ResponseMatrix.from_json(f"ideal_{N_TURNS}turn_orm.json")
with open("name_mapping.json") as fp:
    name_mapping = json.load(fp)
response_matrix.input_names = [name_mapping[name] for name in response_matrix.input_names]

reference_x, reference_y = interface.get_ref_orbit()
reference = np.concatenate((reference_x.flatten(order="F"), reference_y.flatten(order="F")))

before_x, before_y = interface.get_orbit()
print(
    f"RMS before H: {np.std(before_x - reference_x) * 1e6:.1f} um, "
    f"V: {np.std(before_y - reference_y) * 1e6:.1f} um"
)

orbit_correction(
    interface=interface,
    response_matrix=response_matrix,
    reference=reference,
    method="svd_values",
    parameter=64,
    apply=APPLY_CORRECTION,
    plane="H",
)
orbit_correction(
    interface=interface,
    response_matrix=response_matrix,
    reference=reference,
    method="svd_values",
    parameter=64,
    apply=APPLY_CORRECTION,
    plane="V",
)

after_x, after_y = interface.get_orbit()
print(
    f"RMS after H: {np.std(after_x - reference_x) * 1e6:.1f} um, "
    f"V: {np.std(after_y - reference_y) * 1e6:.1f} um"
)
