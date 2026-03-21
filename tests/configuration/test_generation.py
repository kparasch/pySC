"""Tests for pySC.configuration.generation: generate_SC."""
import pytest
import yaml
import os

from pySC.configuration.generation import generate_SC


MACHINE_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "machine_data")
HMBA_MAT = os.path.join(MACHINE_DATA, "hmba.mat")


def _write_minimal_config(tmp_path, **overrides):
    """Write a minimal valid YAML config and return its path.

    The minimal config has a lattice, one magnet category (quads),
    one BPM category, one RF category, and an error table.
    """
    config = {
        "lattice": {
            "lattice_file": HMBA_MAT,
            "simulator": "at",
            "use": "RING",
            "naming": "FamName",
        },
        "error_table": {
            "quad_cal": "0.01",
            "bpm_cal": "0.01",
            "orbit_noise": "1e-4",
            "tbt_noise": "1e-3",
        },
        "magnets": {
            "quadrupoles": {
                "regex": "^Q",
                "components": [{"B2": "quad_cal"}],
            },
        },
        "bpms": {
            "standard": {
                "regex": "^BPM",
                "calibration_error": "bpm_cal",
                "orbit_noise": "orbit_noise",
                "tbt_noise": "tbt_noise",
            },
        },
        "rf": {
            "main": {
                "regex": "^RFC$",
            },
        },
    }
    config.update(overrides)
    filepath = str(tmp_path / "test_config.yaml")
    with open(filepath, "w") as f:
        yaml.dump(config, f)
    return filepath


# ---------------------------------------------------------------------------
# generate_SC from YAML
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_generate_sc_from_yaml(tmp_path):
    """Full generate_SC from a minimal YAML config produces a valid SC."""
    filepath = _write_minimal_config(tmp_path)
    SC = generate_SC(filepath, seed=42)

    # Magnets configured
    assert len(SC.magnet_settings.controls) > 0
    assert "quadrupoles" in SC.magnet_arrays

    # BPMs configured
    assert len(SC.bpm_system.indices) == 10

    # RF configured
    assert "main" in SC.rf_settings.systems

    # Injection configured (from design twiss)
    assert SC.injection.betx > 0


@pytest.mark.slow
def test_generate_sc_missing_lattice_raises(tmp_path):
    """YAML without 'lattice' key raises AssertionError."""
    config = {
        "error_table": {"dummy": "0.001"},
    }
    filepath = str(tmp_path / "bad_config.yaml")
    with open(filepath, "w") as f:
        yaml.dump(config, f)

    with pytest.raises(AssertionError, match="lattice"):
        generate_SC(filepath)


@pytest.mark.slow
def test_generate_sc_scale_errors(tmp_path):
    """scale_errors=2 doubles the error table values."""
    filepath = _write_minimal_config(tmp_path)
    SC = generate_SC(filepath, seed=42, scale_errors=2)

    # After scaling by 2, the error table values should be doubled
    assert float(SC.configuration["error_table"]["quad_cal"]) == pytest.approx(0.02)
    assert float(SC.configuration["error_table"]["bpm_cal"]) == pytest.approx(0.02)


@pytest.mark.slow
def test_generate_sc_sigma_truncate(tmp_path):
    """sigma_truncate=3 sets rng.default_truncation."""
    filepath = _write_minimal_config(tmp_path)
    SC = generate_SC(filepath, seed=42, sigma_truncate=3)

    assert SC.rng.default_truncation == 3
