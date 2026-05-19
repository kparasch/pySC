"""Correct injection over one turn."""

import numpy as np

from pySC import generate_SC

if __name__ == "__main__":
    sc = generate_SC("hmba_config.yaml", seed=1, sigma_truncate=3)

    rng = np.random.default_rng(1)
    sc.magnet_settings.set(sc.tuning.HCORR[0], rng.uniform(-100e-6, 100e-6))
    sc.magnet_settings.set(sc.tuning.VCORR[0], rng.uniform(-100e-6, 100e-6))

    # This loads model_RM/trajectory1.json from the configured model_RM_folder.
    sc.tuning.correct_injection(n_turns=1, method="svd_cutoff", parameter=1e-3)
