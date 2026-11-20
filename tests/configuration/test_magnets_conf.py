"""Tests for pySC.configuration.magnets_conf: configure_magnets."""
import pytest
import numpy as np
import at
from unittest.mock import patch

from pySC.core.lattice import ATLattice
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.configuration.magnets_conf import configure_magnets


def _make_sc_with_magnet_config(hmba_lattice_file, cal_error_sigma="0.01"):
    """Build a fresh SC with a configuration dict that configure_magnets can consume."""
    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")
    config = {
        "error_table": {
            "quad_cal": cal_error_sigma,
            "sext_cal": cal_error_sigma,
        },
        "parameters": {
            "quad_limit": "100",
        },
        "magnets": {
            "quadrupoles": {
                "regex": "^Q",
                "components": [{"B2": "quad_cal"}],
                "limits": [{"B2": "quad_limit"}],
            },
            "sextupoles": {
                "regex": "^S[DF]",
                "components": [
                    {"B3": "sext_cal"},
                    {"B1": "sext_cal"},
                    {"A1": "sext_cal"},
                ],
            },
        },
    }
    SC = SimulatedCommissioning(lattice=lattice, configuration=config, seed=42)
    return SC


# ---------------------------------------------------------------------------
# configure_magnets
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_configure_magnets_creates_controls(hmba_lattice_file):
    """After configure_magnets, SC.magnet_settings.controls is populated."""
    SC = _make_sc_with_magnet_config(hmba_lattice_file)
    configure_magnets(SC)
    assert len(SC.magnet_settings.controls) > 0


@pytest.mark.slow
def test_configure_magnets_creates_magnet_arrays(hmba_lattice_file):
    """SC.magnet_arrays is populated with category -> name mapping."""
    SC = _make_sc_with_magnet_config(hmba_lattice_file)
    configure_magnets(SC)
    assert "quadrupoles" in SC.magnet_arrays
    assert "sextupoles" in SC.magnet_arrays
    assert len(SC.magnet_arrays["quadrupoles"]) > 0
    assert len(SC.magnet_arrays["sextupoles"]) > 0


@pytest.mark.slow
def test_configure_magnets_applies_calibration_errors(hmba_lattice_file):
    """Links have non-unity factors when the error table has non-zero calibration error."""
    SC = _make_sc_with_magnet_config(hmba_lattice_file, cal_error_sigma="0.05")
    configure_magnets(SC)

    # At least one link should have factor != 1
    non_unity = []
    for link_name, link in SC.magnet_settings.links.items():
        if link.error.factor != pytest.approx(1.0):
            non_unity.append(link_name)
            # Factor = 1 + normal(0, 0.05); should be within a reasonable range of 1.0
            assert 0.5 <= link.error.factor <= 1.5, \
                f"Link {link_name} factor={link.error.factor} implausibly far from 1.0 for sigma=0.05"
    assert len(non_unity) > 0, "Expected at least one link with factor != 1"


@pytest.mark.slow
def test_configure_magnets_design_has_unity_factor(hmba_lattice_file):
    """Design magnet settings links always have factor=1."""
    SC = _make_sc_with_magnet_config(hmba_lattice_file, cal_error_sigma="0.05")
    configure_magnets(SC)

    for link_name, link in SC.design_magnet_settings.links.items():
        assert link.error.factor == pytest.approx(1.0), (
            f"Design link {link_name} has factor={link.error.factor}, expected 1.0"
        )


@pytest.mark.slow
@pytest.mark.parametrize("component", ["B1", "B1L"])
@pytest.mark.parametrize("series", [False, True])
def test_configure_magnets_dipole_convention(hmba_lattice_file, component, series):
    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")
    config = {"magnets": {"dipoles": {
        "regex": "^(DQ1B|DQ2C_1)$", "series": series,
        "components": [{component: None}],
    }}}
    SC = SimulatedCommissioning(lattice=lattice, configuration=config, seed=42)
    indices = SC.lattice.find_with_regex("^(DQ1B|DQ2C_1)$")
    assert len(indices) == 2
    # Equal control strengths allow the series case to preserve both magnets.
    initial_setpoint = 0.08
    raw_initial = {}
    for index in indices:
        elem = SC.lattice.design[index]
        scale = elem.Length if component.endswith("L") else 1.0
        elem.PolynomB[0] = initial_setpoint / scale - elem.BendingAngle / elem.Length
        raw_initial[index] = elem.PolynomB[0]
    configure_magnets(SC)
    for settings, use_design in (
        (SC.magnet_settings, False), (SC.design_magnet_settings, True)
    ):
        for control_name in SC.control_arrays["dipoles"]:
            assert settings.get(control_name) == pytest.approx(initial_setpoint)
        for index in indices:
            elem = (SC.lattice.design if use_design else SC.lattice.ring)[index]
            assert elem.PolynomB[0] == pytest.approx(raw_initial[index])
        assert all(link.error.offset == 0 for link in settings.links.values())
        for control_name in SC.control_arrays["dipoles"]:
            settings.set(control_name, 0.12)
        for index in indices:
            elem = (SC.lattice.design if use_design else SC.lattice.ring)[index]
            scale = elem.Length if component.endswith("L") else 1.0
            assert elem.PolynomB[0] == pytest.approx(0.12 / scale - elem.BendingAngle / elem.Length)


@pytest.mark.slow
@pytest.mark.parametrize("component", ["B2", "B2L"])
def test_configure_shifted_quadrupole_feed_down(hmba_lattice_file, component):
    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")
    dip_name = "DQ1B"
    dip_index = 50
    initial_angle = lattice.get_bending_angle(dip_index)
    initial_b0 = lattice.get_magnet_component(dip_index, "B", 0)
    initial_k1 = lattice.get_magnet_component(dip_index, "B", 1)
    length = lattice.get_length(dip_index)
    design_shift = initial_b0 / initial_k1

    config = {
        "error_table": {"quad_cal": "0"},
        "magnets": {
            "shifted_quadrupole": {
                "regex": f"^{dip_name}$",
                "components": [{component: "quad_cal"}],
                "shifted": True,
            },
        },
    }
    SC = SimulatedCommissioning(lattice=lattice, configuration=config, seed=42)
    configure_magnets(SC)

    control_name = f"{dip_name}/{component}"
    feed_down_link = SC.magnet_settings.links[f"{control_name}->{dip_name}/B1"]
    assert feed_down_link.is_integrated is False
    assert SC.lattice.get_bending_angle(dip_index) == pytest.approx(initial_angle)
    assert SC.lattice.get_magnet_component(
        dip_index, "B", 0, use_design=False
    ) == pytest.approx(initial_b0)

    delta_k1 = 0.2
    setpoint = initial_k1 + delta_k1
    if component == "B2L":
        setpoint *= length
    SC.magnet_settings.set(control_name, setpoint)

    assert SC.lattice.get_bending_angle(dip_index) == pytest.approx(initial_angle)
    assert SC.lattice.get_magnet_component(
        dip_index, "B", 1, use_design=False
    ) == pytest.approx(initial_k1 + delta_k1)
    assert SC.lattice.get_magnet_component(
        dip_index, "B", 0, use_design=False
    ) == pytest.approx(initial_b0 + design_shift * delta_k1)


@pytest.mark.slow
def test_configure_magnets_limits(hmba_lattice_file):
    """Controls have limits when configured."""
    SC = _make_sc_with_magnet_config(hmba_lattice_file)
    configure_magnets(SC)

    # Quadrupole controls should have limits
    for control_name in SC.control_arrays["quadrupoles"]:
        control = SC.magnet_settings.controls[control_name]
        assert control.limits is not None, f"{control_name} should have limits set"
        assert control.limits == (-100.0, 100.0)


@pytest.mark.slow
def test_configure_magnets_invert(hmba_lattice_file):
    """Component inversion applies negative factor."""
    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")
    config = {
        "error_table": {"sext_cal": "0"},
        "magnets": {
            "sextupoles": {
                "regex": "^SD1A$",
                "components": [
                    {"B3": "sext_cal"},
                    {"A1": "sext_cal"},
                ],
                "invert": ["A1"],
            },
        },
    }
    SC = SimulatedCommissioning(lattice=lattice, configuration=config, seed=42)
    configure_magnets(SC)

    # Find the A1 link for SD1A
    link_name = "SD1A/A1->SD1A/A1"
    link = SC.magnet_settings.links[link_name]
    # With zero calibration error, factor is normal_trunc(1, 0) = 1.0 then * -1 = -1.0
    assert link.error.factor == pytest.approx(-1.0)


def _make_sc_with_series_config(hmba_lattice_file, component="B2"):
    SC = _make_sc_with_magnet_config(hmba_lattice_file)
    category = SC.configuration["magnets"]["quadrupoles"]
    category.update(
        regex="^QF1[AE]$", series=True,
        components=[{component: "quad_cal"}],
        limits=[{component: "quad_limit"}],
    )
    SC.configuration["magnets"] = {"quadrupoles": category}
    return SC


@pytest.mark.parametrize('trim_component', ['B2', 'B2L', 'A2', 'A4L'])
def test_series_trim_coils_add_independently_and_roundtrip(hmba_lattice_file, tmp_path, trim_component):
    sc = _make_sc_with_series_config(hmba_lattice_file)
    category = sc.configuration['magnets']['quadrupoles']
    category['trim_coils'] = [{trim_component: 'quad_cal'}]
    category['trim_limits'] = [{trim_component: 'trim_limit'}]
    sc.configuration['parameters']['trim_limit'] = '0.1'
    with patch.object(type(sc.rng), 'normal_trunc', side_effect=[1.1, 0.9, 1.2, 0.8]):
        configure_magnets(sc)
    assert sc.control_arrays['quadrupoles'] == [
        'quadrupoles/B2', f'QF1A/{trim_component}', f'QF1E/{trim_component}']
    for settings, elements, shared_factors, trim_factors in (
        (sc.magnet_settings, sc.lattice.ring, [1.1, 0.9], [1.2, 0.8]),
        (sc.design_magnet_settings, sc.lattice.design, [1., 1.], [1., 1.]),
    ):
        settings.set('quadrupoles/B2', 1.5)
        component_type, order = settings.validate_one_component(trim_component)
        for name in ('QF1A', 'QF1E'):
            control = settings.controls[f'{name}/{trim_component}']
            assert control.setpoint == 0.0
            assert control.info.is_trim_coil
            assert control.limits == (-0.1, 0.1)
        settings.set(f'QF1A/{trim_component}', 1.0)  # Clip only this trim.
        for i, name in enumerate(('QF1A', 'QF1E')):
            magnet = settings.magnets[name]
            element = elements[magnet.sim_index]
            base = 1.5 * shared_factors[i] if component_type == 'B' and order == 1 else 0.0
            scale = magnet.length if trim_component.endswith('L') else 1.0
            trim = 0.1 * trim_factors[i] / scale if i == 0 else 0.0
            assert getattr(element, f'Polynom{component_type}')[order] == pytest.approx(base + trim)
        assert settings.get('quadrupoles/B2') == 1.5
        assert settings.controls['quadrupoles/B2'].limits == (-100., 100.)

    path = tmp_path / 'trims.json'
    sc.to_json(path)
    restored = SimulatedCommissioning.from_json(path)
    for key in ('magnet_settings', 'design_magnet_settings'):
        assert getattr(restored, key).model_dump() == getattr(sc, key).model_dump()
        for obj in (sc, restored):
            getattr(obj, key).set(f'QF1E/{trim_component}', -0.03)
        for name in ('QF1A', 'QF1E'):
            original = getattr(sc, key).magnets[name]
            reloaded = getattr(restored, key).magnets[name]
            np.testing.assert_allclose(reloaded.A, original.A)
            np.testing.assert_allclose(reloaded.B, original.B)


def test_higher_order_trim_preserves_existing_strength(hmba_lattice_file):
    sc = _make_sc_with_series_config(hmba_lattice_file)
    sc.configuration['magnets']['quadrupoles']['trim_coils'] = [{'A4': None}]
    index = sc.lattice.find_with_regex('^QF1A$')[0]
    sc.lattice.ensure_max_order(index, 3)
    sc.lattice.design[index].PolynomA[3] = 0.002
    configure_magnets(sc)
    for settings, elements in ((sc.magnet_settings, sc.lattice.ring),
                               (sc.design_magnet_settings, sc.lattice.design)):
        assert elements[index].PolynomA[3] == pytest.approx(0.002)
        settings.set('QF1A/A4', 0.003)
        assert elements[index].PolynomA[3] == pytest.approx(0.005)


@pytest.mark.parametrize('updates,match', [
    ({'series': False, 'trim_coils': [{'B2': None}]}, 'require series=true'),
    ({'trim_coils': [{'B2': None}, {'B2L': None}]}, 'same multipole'),
    ({'trim_coils': [{'B2': None}, {'B2': None}]}, 'same multipole'),
    ({'trim_limits': [{'B2': 'quad_limit'}]}, 'not declared in trim_coils'),
    ({'trim_coils': [{'B2': None}], 'trim_limits': [{'B2': 'missing'}]}, 'was not found'),
    ({'trim_coils': [{'B2': 'missing'}]}, 'not found in error_table'),
    ({'components': [], 'trim_coils': [{'B2': None}]}, 'require shared components'),
])
def test_invalid_trim_configuration(hmba_lattice_file, updates, match):
    sc = _make_sc_with_series_config(hmba_lattice_file)
    sc.configuration['magnets']['quadrupoles'].update(updates)
    with pytest.raises(Exception, match=match):
        configure_magnets(sc)


def test_trim_name_collision(hmba_lattice_file):
    sc = _make_sc_with_series_config(hmba_lattice_file)
    category = sc.configuration['magnets'].pop('quadrupoles')
    category['trim_coils'] = [{'B2': None}]
    sc.configuration['magnets']['QF1A'] = category
    with pytest.raises(ValueError, match='collide'):
        configure_magnets(sc)


@pytest.mark.slow
@pytest.mark.parametrize("component", ["B2", "B2L"])
def test_configure_series_updates_lattice_and_applies_limits(hmba_lattice_file, component):
    SC = _make_sc_with_series_config(hmba_lattice_file, component)
    # Distinct factors catch accidentally configuring only the final link.
    with patch.object(type(SC.rng), "normal_trunc", side_effect=[1.1, 0.9]):
        configure_magnets(SC)

    names = SC.magnet_arrays["quadrupoles"]
    control_name = f"quadrupoles/{component}"
    assert names == ["QF1A", "QF1E"]
    assert SC.control_arrays["quadrupoles"] == [control_name]
    for settings, use_design, factors in (
        (SC.magnet_settings, False, [1.1, 0.9]),
        (SC.design_magnet_settings, True, [1.0, 1.0]),
    ):
        assert settings.controls[control_name].limits == (-100.0, 100.0)
        for requested, expected in [(1.5, 1.5), (101.0, 100.0)]:
            settings.set(control_name, requested)
            assert settings.get(control_name) == pytest.approx(expected)
            for name, factor in zip(names, factors):
                magnet = settings.magnets[name]
                scale = magnet.length if component.endswith("L") else 1.0
                assert SC.lattice.get_magnet_component(
                    magnet.sim_index, "B", 1, use_design=use_design
                ) == pytest.approx(expected * factor / scale)


@pytest.mark.slow
def test_configure_series_empty_selection_creates_no_controls(hmba_lattice_file):
    SC = _make_sc_with_series_config(hmba_lattice_file)
    SC.configuration["magnets"]["quadrupoles"]["regex"] = "^NO_SUCH_MAGNET$"
    configure_magnets(SC)
    assert SC.magnet_arrays["quadrupoles"] == []
    assert SC.control_arrays["quadrupoles"] == []
    for settings in (SC.magnet_settings, SC.design_magnet_settings):
        assert not settings.magnets
        assert not settings.controls
        assert not settings.links


@pytest.mark.slow
def test_configure_series_uses_mean_initial_strength(hmba_lattice_file):
    SC = _make_sc_with_series_config(hmba_lattice_file)
    SC.configuration["error_table"]["quad_cal"] = "0"
    indices = SC.lattice.find_with_regex("^QF1[AE]$")
    assert len(indices) == 2
    for index, strength in zip(indices, [2.0, 4.0]):
        SC.lattice.set_magnet_component(index, strength, "B", 1, use_design=True)
    configure_magnets(SC)
    for settings, use_design in (
        (SC.magnet_settings, False), (SC.design_magnet_settings, True)
    ):
        assert settings.get("quadrupoles/B2") == pytest.approx(3.0)
        for index in indices:
            assert SC.lattice.get_magnet_component(
                index, "B", 1, use_design=use_design
            ) == pytest.approx(3.0)


@pytest.mark.slow
def test_uncontrolled_dipole_strength_and_imperfection_reference(hmba_lattice_file):
    from pySC.core.multipolar_imperfections import ImperfectionsModel, MultipolarImperfectionTable

    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")
    SC = SimulatedCommissioning(lattice=lattice, configuration={"magnets": {
        "dipoles": {"regex": "^DQ1B$", "components": [{"B2": None}]}
    }}, seed=42)
    index = SC.lattice.find_with_regex("^DQ1B$")[0]
    elem = SC.lattice.design[index]
    elem.PolynomB[0] = 0.02
    strength = elem.BendingAngle / elem.Length + 0.02
    configure_magnets(SC)
    magnet = SC.magnet_settings.magnets["DQ1B"]
    assert magnet.offset_B[0] == pytest.approx(strength)
    assert SC.lattice.ring[index].PolynomB[0] == pytest.approx(0.02)
    magnet.imperfections = ImperfectionsModel(list_of_imperfections=[
        MultipolarImperfectionTable(reference_radius=0.01, reference_type=("B", 1),
                                   bn=[100.0], an=[0.0])
    ])
    magnet.update()
    assert SC.lattice.get_magnet_component(index, "B", 0, use_design=False) == pytest.approx(1.01 * strength)
    assert SC.lattice.ring[index].PolynomB[0] == pytest.approx(0.02 + 0.01 * strength)
