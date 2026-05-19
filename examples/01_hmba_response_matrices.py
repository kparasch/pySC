"""Generate HMBA model response matrices."""

from pathlib import Path

if __name__ == "__main__":
    from pySC import generate_SC

    Path("model_RM").mkdir(exist_ok=True)

    sc = generate_SC("hmba_config.yaml", seed=1, sigma_truncate=3)
    sc.tuning.calculate_model_orbit_response_matrix(save_as="model_RM/orbit.json")
    sc.tuning.calculate_model_trajectory_response_matrix(n_turns=1, save_as="model_RM/trajectory1.json")
    sc.tuning.calculate_model_trajectory_response_matrix(n_turns=2, save_as="model_RM/trajectory2.json")
