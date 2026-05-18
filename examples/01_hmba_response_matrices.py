"""Generate HMBA model response matrices."""

import os
from pathlib import Path

examples_path = Path(__file__).resolve().parents[0]

if __name__ == "__main__":
    from pySC import generate_SC

    os.chdir(examples_path)
    Path("output").mkdir(exist_ok=True)

    sc = generate_SC("hmba_config.yaml", seed=1, sigma_truncate=3)
    sc.tuning.calculate_model_orbit_response_matrix(save_as="output/hmba_ideal_orm.json")
    sc.tuning.calculate_model_trajectory_response_matrix(n_turns=1, save_as="output/hmba_ideal_1turn_orm.json")
    sc.tuning.calculate_model_trajectory_response_matrix(n_turns=2, save_as="output/hmba_ideal_2turn_orm.json")
