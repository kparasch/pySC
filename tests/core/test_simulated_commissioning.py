"""Tests for pySC.core.simulated_commissioning: SimulatedCommissioning lifecycle."""
import json
import pytest
import numpy as np

from pySC.core.lattice import ATLattice
from pySC.core.multipolar_imperfections import ImperfectionsModelFactory
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.core.rng import RNG


pytestmark = pytest.mark.slow


@pytest.mark.parametrize("component,shifted,knob", [
    ("B1", False, False), ("B1L", False, False),
    ("B2", True, False), ("B2", False, False),
    ("B1", False, True),
])
def test_legacy_dipole_migration_roundtrip(
    hmba_lattice_file, tmp_path, caplog, component, shifted, knob
):
    from pySC.configuration.magnets_conf import configure_magnets

    sc = SimulatedCommissioning(
        lattice=ATLattice(lattice_file=hmba_lattice_file, naming="FamName"),
        configuration={"magnets": {"dipoles": {
            "regex": "^DQ1B$", "components": [{component: None}],
            "shifted": shifted,
        }}},
    )
    configure_magnets(sc)
    control_name = f"DQ1B/{component}"
    expected = {}
    # Construct a saved legacy state: its summed B1 is the raw AT coefficient.
    for key in ("magnet_settings", "design_magnet_settings"):
        settings = getattr(sc, key)
        if knob:
            settings.add_knob("trim", [control_name], weights=[0.3])
        settings.controls[control_name].setpoint = 0.04
        magnet = settings.magnets["DQ1B"]
        magnet.offset_B[0] = 0.002
        for link in magnet._links:
            link.error.factor *= 1.07
            if link.component == "B" and link.order == 1:
                link.error.offset = -0.01
        expected[key] = magnet.offset_B[0] + sum(
            link.value(settings.controls[link.control_name].setpoint)
            / (magnet.length if link.is_integrated else 1)
            for link in magnet._links
            if link.component == "B" and link.order == 1
        )

    state = sc.model_dump()
    state["version"] = "1.5.6"
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(state))
    migrated = SimulatedCommissioning.from_json(legacy_path)
    assert migrated.version == "1.6.0"
    assert sum("Migrating saved AT dipole" in r.message for r in caplog.records) == 1

    for key, elements in (("magnet_settings", migrated.lattice.ring),
                          ("design_magnet_settings", migrated.lattice.design)):
        settings = getattr(migrated, key)
        magnet = settings.magnets["DQ1B"]
        assert elements[magnet.sim_index].PolynomB[0] == pytest.approx(expected[key])
        assert settings.controls[control_name].setpoint == 0.04
        for name, link in settings.links.items():
            assert link.error.factor == state[key]["links"][name]["error"]["factor"]
        if any(link.component == "B" and link.order == 1 for link in magnet._links):
            assert magnet.offset_B[0] == 0.002

    caplog.clear()
    migrated_path = tmp_path / "migrated.json"
    migrated.to_json(migrated_path)
    reloaded = SimulatedCommissioning.from_json(migrated_path)
    assert not any("Migrating saved AT dipole" in r.message for r in caplog.records)
    for key in ("magnet_settings", "design_magnet_settings"):
        assert getattr(migrated, key).model_dump() == getattr(reloaded, key).model_dump()
        for value in (0.04, 0.06):
            for obj in (migrated, reloaded):
                getattr(obj, key).set(control_name, value)
            use_design = key == "design_magnet_settings"
            ring_a = migrated.lattice.design if use_design else migrated.lattice.ring
            ring_b = reloaded.lattice.design if use_design else reloaded.lattice.ring
            index = getattr(migrated, key).magnets["DQ1B"].sim_index
            np.testing.assert_allclose(ring_a[index].PolynomB, ring_b[index].PolynomB)


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


@pytest.mark.regression
def test_sc_from_json_reloads_lattice_with_extended_multipolar_errors(sc, tmp_path):
    """from_json reloads a fresh lattice and restores extended magnet orders."""
    magnet_name = next(iter(sc.magnet_settings.magnets))
    magnet = sc.magnet_settings.magnets[magnet_name]

    factory = ImperfectionsModelFactory.model_validate(
        {
            "factories": [
                {
                    "reference_radius": 0.0065,
                    "reference_type": ["B", 2],
                    "mean_bn": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "mean_an": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                }
            ]
        }
    )
    magnet.imperfections = factory.create(sc.rng)
    magnet.ensure_max_order(required_max_order=magnet.imperfections.max_order)

    json_file = str(tmp_path / "sc.json")
    sc.to_json(json_file)

    reloaded = SimulatedCommissioning.from_json(json_file)
    reloaded_magnet = reloaded.magnet_settings.magnets[magnet_name]
    reloaded_element = reloaded.lattice.ring[reloaded_magnet.sim_index]

    assert reloaded_magnet.max_order == magnet.imperfections.max_order
    assert len(reloaded_element.PolynomA) >= reloaded_magnet.max_order + 1
    assert len(reloaded_element.PolynomB) >= reloaded_magnet.max_order + 1


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
