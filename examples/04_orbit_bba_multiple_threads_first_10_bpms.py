"""Run orbit BBA with multiple threads for the first 10 BPMs."""

from pySC import generate_SC

if __name__ == "__main__":
    sc = generate_SC("hmba_config.yaml", seed=1, sigma_truncate=3)
    bpm_names = sc.bpm_system.names[:10]

    # This loads model_RM/orbit.json from the configured model_RM_folder.
    sc.tuning.generate_orbit_bba_config(
        max_dx_at_bpm=300e-6,
        max_modulation=20e-6,
    )
    sc.tuning.do_parallel_orbit_bba(
        bpm_names=bpm_names,
        shots_per_orbit=1,
        omp_num_threads=4,
        n_corr_steps=3,
    )
