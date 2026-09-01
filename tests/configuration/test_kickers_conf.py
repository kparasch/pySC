"""Tests for pySC.configuration.kickers_conf."""
from types import SimpleNamespace

import pytest

from pySC.configuration.general import pySCConfigurationError
from pySC.configuration.kickers_conf import configure_kickers
from pySC.core.control import Control, IndivControl
from pySC.core.kickers import ACProgram, KickerSettings, SingleKickProgram, WhiteNoiseProgram


def _make_control(name, magnet_name, component, order):
    return Control(
        name=name,
        setpoint=0.0,
        info=IndivControl(
            magnet_name=magnet_name,
            component=component,
            order=order,
            is_integrated=False,
        ),
    )


def _make_sc(configuration):
    controls = {
        "SF1/B1": _make_control("SF1/B1", "SF1", "B", 1),
        "SF1/A1": _make_control("SF1/A1", "SF1", "A", 1),
        "QF1/B2": _make_control("QF1/B2", "QF1", "B", 2),
    }
    magnet_settings = SimpleNamespace(
        controls=controls,
        magnets={
            "SF1": SimpleNamespace(sim_index=10),
            "QF1": SimpleNamespace(sim_index=20),
        },
    )
    sc = SimpleNamespace(
        configuration=configuration,
        control_arrays={
            "horizontal_correctors": ["SF1/B1"],
            "vertical_correctors": ["SF1/A1"],
            "quadrupoles": ["QF1/B2"],
        },
        magnet_settings=magnet_settings,
        design_magnet_settings=SimpleNamespace(controls=controls),
        kickers=KickerSettings(),
    )
    sc.kickers._parent = sc
    return sc


def test_configure_kickers_creates_programs_without_activating_them():
    sc = _make_sc(
        {
            "kickers": {
                "single_kick": {
                    "single": {
                        "control": [{"horizontal_correctors": "B1"}],
                        "amplitude": "1e-6",
                        "turn_to_kick": "3",
                    },
                },
                "ac": {
                    "shake": {
                        "control": [{"quadrupoles": "B2"}],
                        "amplitude": "2e-6",
                        "tune": "0.25",
                        "ramp_up_turns": "2",
                        "flat_top_turns": "4",
                        "ramp_down_turns": "6",
                    },
                },
                "white_noise": {
                    "noise": {
                        "control": [{"vertical_correctors": "A1"}],
                        "amplitude": "3e-6",
                    },
                },
            },
        }
    )

    configure_kickers(sc)

    assert isinstance(sc.kickers.programs["single"], SingleKickProgram)
    assert sc.kickers.programs["single"].control == "SF1/B1"
    assert sc.kickers.programs["single"].amplitude == pytest.approx(1e-6)
    assert sc.kickers.programs["single"].turn_to_kick == 3

    assert isinstance(sc.kickers.programs["shake"], ACProgram)
    assert sc.kickers.programs["shake"].control == "QF1/B2"
    assert sc.kickers.programs["shake"].amplitude == pytest.approx(2e-6)
    assert sc.kickers.programs["shake"].tune == pytest.approx(0.25)
    assert sc.kickers.programs["shake"].ramp_up_turns == 2
    assert sc.kickers.programs["shake"].flat_top_turns == 4
    assert sc.kickers.programs["shake"].ramp_down_turns == 6

    assert isinstance(sc.kickers.programs["noise"], WhiteNoiseProgram)
    assert sc.kickers.programs["noise"].control == "SF1/A1"
    assert sc.kickers.programs["noise"].amplitude == pytest.approx(3e-6)

    assert sc.kickers.active_programs == []


def test_configure_kickers_rejects_unknown_program_kind():
    sc = _make_sc({"kickers": {"pulse": {}}})

    with pytest.raises(pySCConfigurationError, match="Unknown kicker program kind: pulse"):
        configure_kickers(sc)


def test_configure_kickers_rejects_unknown_program_field():
    sc = _make_sc(
        {
            "kickers": {
                "single_kick": {
                    "single": {
                        "control": [{"horizontal_correctors": "B1"}],
                        "amplitude": 1e-6,
                        "typo": 5,
                    },
                },
            },
        }
    )

    with pytest.raises(pySCConfigurationError, match="kickers/single_kick/single/typo"):
        configure_kickers(sc)


def test_configure_kickers_rejects_missing_mandatory_field():
    sc = _make_sc(
        {
            "kickers": {
                "white_noise": {
                    "noise": {
                        "control": [{"vertical_correctors": "A1"}],
                    },
                },
            },
        }
    )

    with pytest.raises(pySCConfigurationError, match="Mandatory field 'amplitude'"):
        configure_kickers(sc)


def test_configure_kickers_rejects_selector_that_matches_no_controls():
    sc = _make_sc(
        {
            "kickers": {
                "single_kick": {
                    "single": {
                        "control": [{"horizontal_correctors": "A1"}],
                        "amplitude": 1e-6,
                    },
                },
            },
        }
    )

    with pytest.raises(pySCConfigurationError, match="No controls were found"):
        configure_kickers(sc)
