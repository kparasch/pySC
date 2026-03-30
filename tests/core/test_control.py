"""Tests for pySC.core.control: LinearConv, Control, IndivControl, KnobControl, KnobData."""
import pytest

from pySC.core.control import Control, IndivControl, KnobControl, KnobData, LinearConv


# ---------- LinearConv ----------

def test_linear_conv_identity():
    conv = LinearConv()
    assert conv.transform(7.0) == 7.0
    assert conv.transform(-3.5) == -3.5


def test_linear_conv_transform():
    conv = LinearConv(factor=2, offset=3)
    assert conv.transform(5) == 13  # 5 * 2 + 3


# ---------- Control.check_limits_and_set ----------

def test_check_limits_and_set_within_limits():
    ctrl = Control(name="Q1", setpoint=0.0, limits=(-10, 10))
    ctrl.check_limits_and_set(5.0)
    assert ctrl.setpoint == 5.0


def test_check_limits_and_set_below_lower():
    ctrl = Control(name="Q1", setpoint=0.0, limits=(-10, 10))
    ctrl.check_limits_and_set(-20.0)
    assert ctrl.setpoint == -10.0


def test_check_limits_and_set_above_upper():
    ctrl = Control(name="Q1", setpoint=0.0, limits=(-10, 10))
    ctrl.check_limits_and_set(20.0)
    assert ctrl.setpoint == 10.0


def test_check_limits_and_set_no_limits():
    ctrl = Control(name="Q1", setpoint=0.0, limits=None)
    ctrl.check_limits_and_set(1e12)
    assert ctrl.setpoint == 1e12


# ---------- IndivControl ----------

def test_indiv_control_creation():
    ic = IndivControl(magnet_name="QF1", component="B", order=2, is_integrated=True)
    assert ic.magnet_name == "QF1"
    assert ic.component == "B"
    assert ic.order == 2
    assert ic.is_integrated is True


# ---------- KnobControl ----------

def test_knob_control_creation():
    kc = KnobControl(control_names=["Q1", "Q2"], weights=[1.0, -1.0])
    assert kc.control_names == ["Q1", "Q2"]
    assert kc.weights == [1.0, -1.0]


# ---------- KnobData save/load ----------

def test_knob_data_save_load(tmp_path):
    knob = KnobControl(control_names=["Q1", "Q2"], weights=[0.5, 0.5])
    kd = KnobData(data={"my_knob": knob})
    path = tmp_path / "knobs.json"
    kd.save_as(path)
    restored = KnobData.model_validate_json(path.read_text())
    assert "my_knob" in restored.data
    assert restored.data["my_knob"].control_names == ["Q1", "Q2"]
    assert restored.data["my_knob"].weights == [0.5, 0.5]
