"""Correct the closed orbit using an operational interface."""

import json

import numpy as np

from interface import Interface
from pySC import ResponseMatrix, orbit_correction


APPLY_CORRECTION = True

interface = Interface()

response_matrix = ResponseMatrix.from_json("ideal_orm.json")
with open("name_mapping.json") as fp:
    name_mapping = json.load(fp)
response_matrix.input_names = [name_mapping[name] for name in response_matrix.input_names]

reference_x, reference_y = interface.get_ref_orbit()
reference = np.concatenate((reference_x.flatten(order="F"), reference_y.flatten(order="F")))

trims = orbit_correction(
    interface=interface,
    response_matrix=response_matrix,
    reference=reference,
    method="svd_cutoff",
    parameter=1e-3,
    apply=APPLY_CORRECTION,
)
print(trims)

micado_trims = orbit_correction(
    interface=interface,
    response_matrix=response_matrix,
    reference=reference,
    method="micado",
    parameter=1,
    apply=False,
)
print(f"Micado trims: {micado_trims}")
