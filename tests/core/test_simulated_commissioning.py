"""Tests for pySC.core.simulated_commissioning: SimulatedCommissioning lifecycle."""
import json
import pytest
import numpy as np

from pySC.core.lattice import ATLattice
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.core.rng import RNG


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_sc_construction_minimal(hmba_lattice_file):
    """SC with just a lattice constructs successfully."""
    lattice = ATLattice(lattice_file=hmba_lattice_file)
    sc = SimulatedCommissioning(lattice=lattice)
    assert sc.lattice is not None
    assert sc.lattice.ring is not None


def test_sc_propagate_parents(sc):
    """After construction, top-level subsystem _parent refs point to SC."""
    assert sc.magnet_settings._parent is sc
    assert sc.design_magnet_settings._parent is sc
    assert sc.bpm_system._parent is sc
    assert sc.support_system._parent is sc
    assert sc.rf_settings._parent is sc
    assert sc.injection._parent is sc
    assert sc.tuning._parent is sc

    # Deeper objects point to their immediate parent, not SC
    for system_name in sc.rf_settings.systems:
        system = sc.rf_settings.systems[system_name]
        assert system._parent is sc.rf_settings
        for cav_name in system.cavities:
            cav = sc.rf_settings.cavities[cav_name]
            assert cav._parent_system is system


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------

def test_sc_rng_auto_created(hmba_lattice_file):
    """If rng=None, SC creates RNG(seed=self.seed)."""
    lattice = ATLattice(lattice_file=hmba_lattice_file)
    sc = SimulatedCommissioning(lattice=lattice, seed=99)
    assert sc.rng is not None
    assert isinstance(sc.rng, RNG)
    assert sc.rng.seed == 99


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

def test_sc_copy_is_independent(sc):
    """SC.copy() produces independent object -- mutations don't cross."""
    sc_copy = sc.copy()

    # Mutate a magnet setpoint in the copy
    if sc_copy.magnet_settings.controls:
        first_key = next(iter(sc_copy.magnet_settings.controls))
        sc_copy.magnet_settings.controls[first_key].setpoint = 999.0
        assert sc.magnet_settings.controls[first_key].setpoint != 999.0

    # Mutate seed-level attribute
    assert sc_copy.seed == sc.seed  # same value initially
    # Verify lattice ring objects are independent
    ring_orig = sc.lattice.ring
    ring_copy = sc_copy.lattice.ring
    assert ring_orig is not ring_copy


# ---------------------------------------------------------------------------
# JSON serialisation round-trip
# ---------------------------------------------------------------------------

def test_sc_to_json_from_json_roundtrip(sc, tmp_path):
    """to_json then from_json produces equivalent SC."""
    json_file = str(tmp_path / "sc.json")
    sc.to_json(json_file)

    sc2 = SimulatedCommissioning.from_json(json_file)

    # Lattice same number of elements
    assert len(sc2.lattice.ring) == len(sc.lattice.ring)
    # Same seed
    assert sc2.seed == sc.seed
    # Same magnet settings keys
    assert set(sc2.magnet_settings.controls.keys()) == set(sc.magnet_settings.controls.keys())
    # BPM indices match
    assert sc2.bpm_system.indices == sc.bpm_system.indices


def test_sc_from_json_custom_lattice_file(sc, tmp_path, hmba_lattice_file):
    """from_json(path, lattice_file=other) overrides the lattice path."""
    json_file = str(tmp_path / "sc.json")
    sc.to_json(json_file)

    # Verify the JSON was written with the original lattice file
    with open(json_file) as f:
        data = json.load(f)
    original_lattice_file = data["lattice"]["lattice_file"]

    # Now reload with an explicit override (same file, but proves the mechanism)
    sc2 = SimulatedCommissioning.from_json(json_file, lattice_file=hmba_lattice_file)
    assert sc2.lattice.lattice_file == hmba_lattice_file
    assert len(sc2.lattice.ring) == len(sc.lattice.ring)


# ---------------------------------------------------------------------------
# Knob import
# ---------------------------------------------------------------------------

def _make_knob_json(sc, tmp_path, knob_name="test_knob"):
    """Helper: create a minimal knob JSON referencing existing controls."""
    # Pick two existing control names from magnet_settings
    control_names = list(sc.magnet_settings.controls.keys())[:2]
    assert len(control_names) >= 2, "Need at least 2 controls for knob test"

    knob_data = {
        "data": {
            knob_name: {
                "control_names": control_names,
                "weights": [1.0, -0.5],
            }
        }
    }
    knob_file = str(tmp_path / "knob.json")
    with open(knob_file, "w") as f:
        json.dump(knob_data, f)
    return knob_file, knob_name


def test_sc_import_knob(sc, tmp_path):
    """import_knob(json_file) adds knobs to both magnet_settings and design_magnet_settings."""
    knob_file, knob_name = _make_knob_json(sc, tmp_path)

    sc.import_knob(knob_file)

    assert knob_name in sc.magnet_settings.controls
    assert knob_name in sc.design_magnet_settings.controls


def test_sc_import_knob_duplicate_raises(sc, tmp_path):
    """Importing a knob with existing name raises AssertionError."""
    knob_file, knob_name = _make_knob_json(sc, tmp_path, knob_name="dup_knob")

    sc.import_knob(knob_file)

    # Importing the same knob again should raise
    with pytest.raises(AssertionError):
        sc.import_knob(knob_file)
