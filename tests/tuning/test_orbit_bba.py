"""Tests for pySC.tuning.orbit_bba: orbit beam-based alignment."""
import pytest
import numpy as np
from types import SimpleNamespace

from pySC.apps.response_matrix import ResponseMatrix
from pySC.core.control import IndivControl
from pySC.tuning.orbit_bba import (
    Orbit_BBA_Configuration,
    orbit_bba,
    get_mag_s_pos,
)


pytestmark = pytest.mark.slow


@pytest.fixture
def sc_with_orbit_rm(sc_tuning):
    """SC with pre-calculated orbit response matrix (needed for BBA config)."""
    sc = sc_tuning
    sc.tuning.calculate_model_orbit_response_matrix()
    return sc


@pytest.mark.xfail(reason="HMBA single-cell ring has near-zero response in some BPM/corrector pairs")
def test_generate_orbit_bba_config(sc_with_orbit_rm):
    """Orbit_BBA_Configuration.generate_config runs and produces config for each BPM."""
    sc = sc_with_orbit_rm
    config = Orbit_BBA_Configuration.generate_config(
        SC=sc,
        max_dx_at_bpm=0.3e-3,
        max_modulation=20e-6,
    )
    assert isinstance(config, Orbit_BBA_Configuration)
    # Should have one entry per BPM
    assert len(config.config) == len(sc.bpm_system.names)
    for bpm_name in sc.bpm_system.names:
        assert bpm_name in config.config
        entry = config.config[bpm_name]
        assert 'QUAD' in entry
        assert 'HCORR' in entry
        assert 'VCORR' in entry
        assert 'QUAD_dk_H' in entry
        assert 'QUAD_dk_V' in entry


def _sextupole_b3_controls(sc):
    """Return normal sextupole controls registered on the test SC."""
    return [
        name for name, control in sc.magnet_settings.controls.items()
        if control.info.component == "B" and control.info.order == 3
    ]


def _make_mock_orbit_config_sc(bba_magnets):
    """Create a minimal SC-like object for deterministic orbit BBA config tests."""
    controls = {
        "q1/B2": SimpleNamespace(info=IndivControl(magnet_name="q1", component="B", order=2, is_integrated=False)),
        "s1/B3": SimpleNamespace(info=IndivControl(magnet_name="s1", component="B", order=3, is_integrated=False)),
        "ch1/B1": SimpleNamespace(info=IndivControl(magnet_name="ch1", component="B", order=1, is_integrated=True)),
        "ch2/B1": SimpleNamespace(info=IndivControl(magnet_name="ch2", component="B", order=1, is_integrated=True)),
        "cv1/A1": SimpleNamespace(info=IndivControl(magnet_name="cv1", component="A", order=1, is_integrated=True)),
        "cv2/A1": SimpleNamespace(info=IndivControl(magnet_name="cv2", component="A", order=1, is_integrated=True)),
    }
    magnets = {
        "q1": SimpleNamespace(sim_index=1, length=0.5),
        "s1": SimpleNamespace(sim_index=2, length=0.4),
        "ch1": SimpleNamespace(sim_index=0, length=0.1),
        "ch2": SimpleNamespace(sim_index=2, length=0.1),
        "cv1": SimpleNamespace(sim_index=0, length=0.1),
        "cv2": SimpleNamespace(sim_index=2, length=0.1),
    }
    matrix = np.array([
        [1.0, 0.5, 0.0, 0.0],
        [0.7, 1.2, 0.0, 0.0],
        [0.0, 0.0, 1.1, 0.6],
        [0.0, 0.0, 0.8, 1.3],
    ])
    response_matrix = ResponseMatrix(
        matrix=matrix,
        input_names=["ch1/B1", "ch2/B1", "cv1/A1", "cv2/A1"],
        output_names=["bpm1", "bpm2", "bpm1", "bpm2"],
        input_planes=["H", "H", "V", "V"],
        output_planes=["H", "H", "V", "V"],
    )
    tuning = SimpleNamespace(
        bba_magnets=bba_magnets,
        HCORR=["ch1/B1", "ch2/B1"],
        VCORR=["cv1/A1", "cv2/A1"],
        response_matrix={"orbit": response_matrix},
        fetch_response_matrix=lambda *args, **kwargs: None,
    )
    return SimpleNamespace(
        tuning=tuning,
        magnet_settings=SimpleNamespace(controls=controls, magnets=magnets),
        bpm_system=SimpleNamespace(indices=[3, 4], names=["bpm1", "bpm2"]),
        lattice=SimpleNamespace(twiss={
            "s": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            "betx": np.ones(5),
            "bety": np.ones(5),
            "qx": 0.31,
            "qy": 0.32,
        }),
    )


def test_generate_orbit_bba_config_sextupole_unit():
    """Orbit BBA config supports normal sextupole BBA magnets."""
    sc = _make_mock_orbit_config_sc(["s1/B3"])

    config = Orbit_BBA_Configuration.generate_config(
        SC=sc,
        max_dx_at_bpm=0.3e-3,
        max_modulation=20e-6,
        max_dx_at_bpm_sextupole=1e-3,
        max_modulation_sextupole=10e-6,
    )

    assert {entry["magnet_type"] for entry in config.config.values()} == {"normal_sextupole"}
    for entry in config.config.values():
        assert entry["QUAD"] == "s1/B3"
        assert np.isfinite(entry["QUAD_dk_H"])
        assert np.isfinite(entry["QUAD_dk_V"])


def test_generate_orbit_bba_config_ignore_sextupoles_unit():
    """ignore_sextupoles removes normal sextupole BBA magnets."""
    sc = _make_mock_orbit_config_sc(["s1/B3", "q1/B2"])

    config = Orbit_BBA_Configuration.generate_config(SC=sc, ignore_sextupoles=True)

    assert {entry["QUAD"] for entry in config.config.values()} == {"q1/B2"}
    assert "normal_sextupole" not in {entry["magnet_type"] for entry in config.config.values()}


def test_generate_orbit_bba_config_empty_filter_unit():
    """ignore_sextupoles fails clearly when all BBA magnets are sextupoles."""
    sc = _make_mock_orbit_config_sc(["s1/B3"])

    with pytest.raises(AssertionError, match="No BBA magnets available"):
        Orbit_BBA_Configuration.generate_config(SC=sc, ignore_sextupoles=True)


@pytest.mark.xfail(reason="HMBA single-cell ring has near-zero response in some BPM/corrector pairs")
def test_generate_orbit_bba_config_supports_sextupole_magnets(sc_with_orbit_rm):
    """Orbit BBA config labels normal sextupole BBA magnets."""
    sc = sc_with_orbit_rm
    original_bba_magnets = sc.tuning.bba_magnets
    try:
        sc.tuning.bba_magnets = _sextupole_b3_controls(sc)

        config = Orbit_BBA_Configuration.generate_config(
            SC=sc,
            max_dx_at_bpm=0.3e-3,
            max_modulation=20e-6,
            max_dx_at_bpm_sextupole=1e-3,
            max_modulation_sextupole=10e-6,
        )
    finally:
        sc.tuning.bba_magnets = original_bba_magnets

    assert {entry["magnet_type"] for entry in config.config.values()} == {"normal_sextupole"}


@pytest.mark.xfail(reason="HMBA single-cell ring has near-zero response in some BPM/corrector pairs")
def test_generate_orbit_bba_config_ignore_sextupoles(sc_with_orbit_rm):
    """ignore_sextupoles removes normal sextupole BBA magnets from config selection."""
    sc = sc_with_orbit_rm
    original_bba_magnets = sc.tuning.bba_magnets
    try:
        sc.tuning.bba_magnets = _sextupole_b3_controls(sc) + original_bba_magnets

        config = Orbit_BBA_Configuration.generate_config(
            SC=sc,
            ignore_sextupoles=True,
        )
    finally:
        sc.tuning.bba_magnets = original_bba_magnets

    assert "normal_sextupole" not in {entry["magnet_type"] for entry in config.config.values()}


def test_generate_orbit_bba_config_raises_when_filter_leaves_no_magnets(sc_with_orbit_rm):
    """ignore_sextupoles fails clearly when only sextupole BBA magnets exist."""
    sc = sc_with_orbit_rm
    original_bba_magnets = sc.tuning.bba_magnets
    try:
        sc.tuning.bba_magnets = _sextupole_b3_controls(sc)

        with pytest.raises(AssertionError, match="No BBA magnets available"):
            Orbit_BBA_Configuration.generate_config(
                SC=sc,
                ignore_sextupoles=True,
            )
    finally:
        sc.tuning.bba_magnets = original_bba_magnets


@pytest.mark.xfail(reason="HMBA single-cell ring has near-zero response in some BPM/corrector pairs")
def test_orbit_bba_single_bpm(sc_with_orbit_rm):
    """orbit_bba runs without error on a single BPM."""
    sc = sc_with_orbit_rm
    bpm_name = sc.bpm_system.names[0]

    # Run orbit BBA for horizontal plane
    offset, offset_err = orbit_bba(
        sc, bpm_name, plane='H', shots_per_orbit=1, n_corr_steps=3,
    )
    assert isinstance(offset, float)
    assert isinstance(offset_err, float)


def test_orbit_bba_get_mag_s_pos(sc_with_orbit_rm):
    """get_mag_s_pos returns s-positions for magnet controls."""
    sc = sc_with_orbit_rm
    mag_controls = sc.tuning.HCORR[:2]
    s_positions = get_mag_s_pos(sc, mag_controls)
    assert len(s_positions) == 2
    for s in s_positions:
        assert isinstance(s, float)
        assert s >= 0
