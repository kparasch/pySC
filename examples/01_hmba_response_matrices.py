"""Generate HMBA model response matrices for later examples."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
OUTPUT_DIR = EXAMPLES_DIR / "output"
CONFIG_PATH = EXAMPLES_DIR / "hmba_config.yaml"
RESOLVED_CONFIG_PATH = OUTPUT_DIR / "hmba_config.resolved.yaml"
CACHE_DIR = OUTPUT_DIR / ".cache"


def write_resolved_config() -> Path:
    """Write a generated config with absolute paths for cwd-independent runs."""
    with CONFIG_PATH.open() as stream:
        config = yaml.safe_load(stream)

    lattice_file = (EXAMPLES_DIR / config["lattice"]["lattice_file"]).resolve()
    config["lattice"]["lattice_file"] = str(lattice_file)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with RESOLVED_CONFIG_PATH.open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    return RESOLVED_CONFIG_PATH


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))
    sys.path.insert(0, str(REPO_ROOT))

    from pySC import generate_SC

    config_path = write_resolved_config()
    sc = generate_SC(str(config_path), seed=1, sigma_truncate=3)

    orbit_rm = OUTPUT_DIR / "hmba_ideal_orm.json"
    trajectory_1turn_rm = OUTPUT_DIR / "hmba_ideal_1turn_orm.json"
    trajectory_2turn_rm = OUTPUT_DIR / "hmba_ideal_2turn_orm.json"

    sc.tuning.calculate_model_orbit_response_matrix(save_as=str(orbit_rm))
    sc.tuning.calculate_model_trajectory_response_matrix(
        n_turns=1,
        save_as=str(trajectory_1turn_rm),
    )
    sc.tuning.calculate_model_trajectory_response_matrix(
        n_turns=2,
        save_as=str(trajectory_2turn_rm),
    )

    print("Wrote response matrices:")
    print(f"  {orbit_rm.relative_to(REPO_ROOT)}")
    print(f"  {trajectory_1turn_rm.relative_to(REPO_ROOT)}")
    print(f"  {trajectory_2turn_rm.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
