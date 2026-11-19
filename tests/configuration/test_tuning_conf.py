"""Tests for pySC.configuration.tuning_conf: configure_tuning."""
import pytest

from pySC.core.lattice import ATLattice
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.configuration.magnets_conf import configure_magnets
from pySC.configuration.tuning_conf import configure_tuning, resolve_controls


def _make_sc_with_tuning_config(hmba_lattice_file):
    """Build a fresh SC with magnets and tuning configuration."""
    lattice = ATLattice(lattice_file=hmba_lattice_file, naming="FamName")
    config = {
        "error_table": {
            "sext_cal": "0",
        },
        "magnets": {
            "sextupoles": {
                "regex": "^S[DF]",
                "components": [
                    {"B3": "sext_cal"},
                    {"B1": "sext_cal"},
                    {"A1": "sext_cal"},
                ],
            },
            "quadrupoles": {
                "regex": "^Q",
                "components": [{"B2": "sext_cal"}],
            },
        },
        "tuning": {
            "HCORR": [
                {"sextupoles": "B1"},
            ],
            "VCORR": [
                {"sextupoles": "A1"},
            ],
            "multipoles": [
                {"sextupoles": "B3"},
            ],
            "bba_magnets": [
                {"quadrupoles": "B2"},
            ],
            "tune": {
                "controls_1": {
                    "regex": "^QF1",
                    "component": "B2",
                },
                "controls_2": {
                    "regex": "^QD2",
                    "component": "B2",
                },
            },
        },
    }
    SC = SimulatedCommissioning(lattice=lattice, configuration=config, seed=42)
    return SC


# ---------------------------------------------------------------------------
# configure_tuning
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_configure_tuning(hmba_lattice_file):
    """configure_tuning populates HCORR, VCORR, multipoles, bba_magnets, and tune controls."""
    SC = _make_sc_with_tuning_config(hmba_lattice_file)
    configure_magnets(SC)
    configure_tuning(SC)

    # HCORR should contain sextupole B1 corrector controls
    assert len(SC.tuning.HCORR) > 0
    for name in SC.tuning.HCORR:
        assert "/B1" in name

    # VCORR should contain sextupole A1 corrector controls
    assert len(SC.tuning.VCORR) > 0
    for name in SC.tuning.VCORR:
        assert "/A1" in name

    # multipoles should contain sextupole B3 controls
    assert len(SC.tuning.multipoles) > 0
    for name in SC.tuning.multipoles:
        assert "/B3" in name

    # bba_magnets should contain quadrupole B2 controls
    assert len(SC.tuning.bba_magnets) > 0
    for name in SC.tuning.bba_magnets:
        assert "/B2" in name

    # tune controls should be populated
    assert len(SC.tuning.tune.controls_1) > 0
    assert len(SC.tuning.tune.controls_2) > 0
    for name in SC.tuning.tune.controls_1:
        assert "/B2" in name
    for name in SC.tuning.tune.controls_2:
        assert "/B2" in name


@pytest.mark.parametrize("form", ["list", "regex", "mapping"])
def test_resolve_controls_shared_and_integrated(hmba_lattice_file, tmp_path, form):
    sc = _make_sc_with_tuning_config(hmba_lattice_file)
    sc.configuration['magnets']['quadrupoles'].update(
        series=True, components=[{'B2': None}, {'A2L': None}])
    configure_magnets(sc)
    sc.magnet_settings.add_knob('skew_knob', ['quadrupoles/A2L'])
    if form == 'list':
        selection = [{'quadrupoles': 'A2L'}, {'quadrupoles': 'A2L'}]
    elif form == 'regex':
        # A subset of a series still resolves to its single shared control.
        selection = {'regex': '^QF1', 'component': 'A2L'}
    else:
        path = tmp_path / 'mapping.yaml'
        magnet = sc.magnet_settings.magnets['QF1A']
        path.write_text(f'QF1A: {magnet.sim_index}\n')
        selection = {'mapping': str(path), 'component': 'A2L'}
    assert resolve_controls(sc, 'c_minus/controls', selection) == ['quadrupoles/A2L']
    assert resolve_controls(sc, 'c_minus/controls', {'regex': '^QF1', 'component': 'A2'}) == []


def test_resolve_controls_deduplicates_and_sorts(hmba_lattice_file):
    sc = _make_sc_with_tuning_config(hmba_lattice_file)
    configure_magnets(sc)
    sc.control_arrays['reversed'] = list(reversed(sc.control_arrays['quadrupoles']))
    expected = sc.control_arrays['quadrupoles']
    assert resolve_controls(sc, 'tune/controls_1', [{'reversed': 'B2'}, {'quadrupoles': 'B2'}]) == expected
    assert resolve_controls(sc, 'tune/controls_1', {'regex': '^Q', 'component': 'B2'}) == expected


@pytest.mark.parametrize('selection', [
    {'regex': '^Q'}, {'component': 'B2'},
    {'regex': '^Q', 'mapping': 'unused.yaml', 'component': 'B2'},
    [{'quadrupoles': 'B2', 'sextupoles': 'B3'}],
    {'components': {'quadrupoles': 'B2'}},
    {'components': [{'quadrupoles': 'B2'}], 'regex': '^Q'},
    {'components': [{'quadrupoles': 'B2'}], 'component': 'B2'},
    {'regex': '^Q', 'component': 'B2', 'use_trim_coils': 'false'},
])
def test_resolve_controls_rejects_invalid_selection(hmba_lattice_file, selection):
    sc = _make_sc_with_tuning_config(hmba_lattice_file)
    with pytest.raises(ValueError):
        resolve_controls(sc, 'tune/controls_1', selection)


@pytest.mark.parametrize('form', ['components', 'regex', 'mapping'])
@pytest.mark.parametrize('use_trim_coils', [None, False, True])
def test_resolve_controls_selects_main_or_trim_coils(hmba_lattice_file, tmp_path, form, use_trim_coils):
    sc = _make_sc_with_tuning_config(hmba_lattice_file)
    sc.configuration['magnets']['quadrupoles'].update(
        regex='^QF1[AE]$', series=True,
        components=[{'B2': None}], trim_coils=[{'B2': None}, {'A2L': None}])
    configure_magnets(sc)
    sc.magnet_settings.add_knob('trim_knob', ['QF1A/B2'])
    if form == 'components':
        config = {'components': [{'quadrupoles': 'B2'}, {'quadrupoles': 'B2'}]}
        expected_trims = ['QF1A/B2', 'QF1E/B2']
    elif form == 'regex':
        config = {'regex': '^QF1A$', 'component': 'B2'}
        expected_trims = ['QF1A/B2']
    else:
        path = tmp_path / 'trims.yaml'
        path.write_text(''.join(f'{name}: {sc.magnet_settings.magnets[name].sim_index}\n'
                                for name in ('QF1E', 'QF1A')))
        config = {'mapping': str(path), 'component': 'B2'}
        expected_trims = ['QF1A/B2', 'QF1E/B2']
    if use_trim_coils is not None:
        config['use_trim_coils'] = use_trim_coils
    assert resolve_controls(sc, 'tune/controls_1', config) == (
        expected_trims if use_trim_coils else ['quadrupoles/B2'])
    # Existing bare lists continue to exclude trims.
    assert resolve_controls(sc, 'tune/controls_1', [{'quadrupoles': 'B2'}]) == ['quadrupoles/B2']


@pytest.mark.parametrize('form', ['components', 'regex'])
def test_use_trim_coils_has_no_fallback_and_matches_integrated_component(hmba_lattice_file, form):
    sc = _make_sc_with_tuning_config(hmba_lattice_file)
    sc.configuration['magnets']['quadrupoles'].update(
        series=True, trim_coils=[{'A2L': None}])
    configure_magnets(sc)

    def select(group, regex, component):
        config = {'components': [{group: component}]} if form == 'components' else {
            'regex': regex, 'component': component}
        return resolve_controls(sc, 'c_minus/controls', dict(config, use_trim_coils=True))

    assert select('quadrupoles', '^Q', 'B2') == []  # Shared control is not a trim.
    assert select('sextupoles', '^S[DF]', 'B3') == [
        f'{name}/B3' for name in sc.magnet_arrays['sextupoles']]
    assert select('quadrupoles', '^Q', 'A2') == []
    expected = [f'{name}/A2L' for name in sc.magnet_arrays['quadrupoles']]
    assert select('quadrupoles', '^Q', 'A2L') == expected


@pytest.mark.parametrize('form', ['components', 'regex', 'mapping'])
@pytest.mark.parametrize('use_trim_coils', [None, False, True])
def test_hcorr_mixes_individual_and_series_magnets(hmba_lattice_file, tmp_path, form, use_trim_coils):
    sc = _make_sc_with_tuning_config(hmba_lattice_file)
    sc.configuration['magnets']['quadrupoles'].update(
        regex='^QF1[AE]$', series=True,
        components=[{'B1': None}], trim_coils=[{'B1': None}])
    configure_magnets(sc)
    if form == 'components':
        config = {'components': [{'sextupoles': 'B1'}, {'quadrupoles': 'B1'}]}
    elif form == 'regex':
        config = {'regex': '^S[DF]|^QF1[AE]$', 'component': 'B1'}
    else:
        path = tmp_path / 'mixed.yaml'
        path.write_text(''.join(f'{name}: {magnet.sim_index}\n'
                               for name, magnet in sc.magnet_settings.magnets.items()))
        config = {'mapping': str(path), 'component': 'B1'}
    if use_trim_coils is not None:
        config['use_trim_coils'] = use_trim_coils
    sc.configuration['tuning'] = {'HCORR': config}
    configure_tuning(sc)
    expected = {f'{name}/B1' for name in sc.magnet_arrays['sextupoles']}
    expected.update(['QF1A/B1', 'QF1E/B1'] if use_trim_coils else ['quadrupoles/B1'])
    assert set(sc.tuning.HCORR) == expected
    assert len(sc.tuning.HCORR) == len(expected)
