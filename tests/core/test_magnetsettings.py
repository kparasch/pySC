"""Tests for pySC.core.magnetsettings: MagnetSettings."""
import pytest
from unittest.mock import MagicMock

from pySC.core.magnetsettings import MagnetSettings
from pySC.core.magnet import Magnet, ControlMagnetLink
from pySC.core.control import Control, LinearConv


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_settings():
    """Return a MagnetSettings wired to a mock SC so update() can propagate."""
    ms = MagnetSettings()
    mock_lattice = MagicMock()
    mock_sc = MagicMock()
    mock_sc.lattice = mock_lattice
    ms._parent = mock_sc
    return ms, mock_sc


def _add_simple_magnet(ms, name="m1", sim_index=0, max_order=1, length=1.0):
    """Add a bare magnet to *ms* and wire its parent."""
    m = Magnet(name=name, sim_index=sim_index, max_order=max_order, length=length)
    m._parent = ms
    ms.magnets[name] = m
    ms.index_mapping[sim_index] = name
    return m


def _add_simple_control(ms, name="c1", setpoint=0.0):
    ctrl = Control(name=name, setpoint=setpoint)
    ms.controls[name] = ctrl
    return ctrl


def _wire_link(ms, link_name="lk1", magnet_name="m1", control_name="c1",
               component="B", order=1, **kwargs):
    link = ControlMagnetLink(
        link_name=link_name, magnet_name=magnet_name, control_name=control_name,
        component=component, order=order, **kwargs,
    )
    ms.links[link_name] = link
    return link


# ---------------------------------------------------------------------------
# add_magnet / add_control / add_link validation
# ---------------------------------------------------------------------------

def test_add_magnet_duplicate_raises():
    ms, _ = _make_settings()
    _add_simple_magnet(ms, name="dup", sim_index=0)
    with pytest.raises(ValueError, match="already exists"):
        ms.add_magnet(Magnet(name="dup", sim_index=1, max_order=1))


def test_add_control_duplicate_raises():
    ms, _ = _make_settings()
    _add_simple_control(ms, name="dup")
    with pytest.raises(ValueError, match="already exists"):
        ms.add_control(Control(name="dup", setpoint=0.0))


def test_add_link_unknown_magnet_raises():
    ms, _ = _make_settings()
    _add_simple_control(ms, name="c1")
    link = ControlMagnetLink(
        link_name="lk", magnet_name="no_such_magnet", control_name="c1",
        component="B", order=1,
    )
    with pytest.raises(ValueError, match="not found"):
        ms.add_link(link)


def test_add_link_unknown_control_raises():
    ms, _ = _make_settings()
    _add_simple_magnet(ms, name="m1")
    link = ControlMagnetLink(
        link_name="lk", magnet_name="m1", control_name="no_such_control",
        component="B", order=1,
    )
    with pytest.raises(ValueError, match="not found"):
        ms.add_link(link)


# ---------------------------------------------------------------------------
# set / get
# ---------------------------------------------------------------------------

def test_set_get_roundtrip():
    ms, _ = _make_settings()
    m = _add_simple_magnet(ms)
    ctrl = _add_simple_control(ms, name="c1", setpoint=0.0)
    link = _wire_link(ms)
    ms.connect_links()

    ms.set("c1", 3.14)
    assert ms.get("c1") == pytest.approx(3.14)


def test_set_unknown_control_raises():
    ms, _ = _make_settings()
    with pytest.raises(ValueError, match="not found"):
        ms.set("nonexistent", 1.0)


def test_set_triggers_magnet_update():
    """After set(), the magnet's A/B arrays reflect the new setpoint."""
    ms, _ = _make_settings()
    m = _add_simple_magnet(ms, max_order=1, length=1.0)
    _add_simple_control(ms, name="c1")
    _wire_link(ms, component="B", order=1)
    ms.connect_links()

    ms.set("c1", 7.0)
    # B[0] = offset(0) + link_value(7.0)
    assert m.B[0] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# get_many / set_many
# ---------------------------------------------------------------------------

def test_get_many_set_many():
    ms, _ = _make_settings()
    _add_simple_magnet(ms, name="m1", sim_index=0)
    _add_simple_control(ms, name="c1")
    _wire_link(ms, link_name="lk1", magnet_name="m1", control_name="c1")

    _add_simple_magnet(ms, name="m2", sim_index=1)
    _add_simple_control(ms, name="c2")
    _wire_link(ms, link_name="lk2", magnet_name="m2", control_name="c2")

    ms.connect_links()

    ms.set_many({"c1": 1.5, "c2": 2.5})
    result = ms.get_many(["c1", "c2"])
    assert result == {"c1": pytest.approx(1.5), "c2": pytest.approx(2.5)}


# ---------------------------------------------------------------------------
# connect_links idempotent
# ---------------------------------------------------------------------------

def test_connect_links_idempotent():
    ms, _ = _make_settings()
    m = _add_simple_magnet(ms)
    _add_simple_control(ms)
    _wire_link(ms)

    ms.connect_links()
    first_magnet_links = list(m._links)
    first_ctrl_links = list(ms.controls["c1"]._links)

    ms.connect_links()
    assert len(m._links) == len(first_magnet_links)
    assert len(ms.controls["c1"]._links) == len(first_ctrl_links)


# ---------------------------------------------------------------------------
# validate_one_component
# ---------------------------------------------------------------------------

def test_validate_one_component_valid():
    ms, _ = _make_settings()
    comp_type, order = ms.validate_one_component("B2")
    assert comp_type == "B"
    assert order == 1  # order = int("2") - 1

    comp_type, order = ms.validate_one_component("A1L")
    assert comp_type == "A"
    assert order == 0  # order = int("1") - 1


def test_validate_one_component_invalid():
    ms, _ = _make_settings()
    with pytest.raises(ValueError, match="must start with"):
        ms.validate_one_component("X1")
    with pytest.raises(ValueError, match="must be a positive integer"):
        ms.validate_one_component("B0")


# ---------------------------------------------------------------------------
# add_individually_powered_magnet
# ---------------------------------------------------------------------------

def test_add_individually_powered_magnet():
    """Creates magnet, controls, and links correctly."""
    ms, mock_sc = _make_settings()
    mock_sc.lattice.get_magnet_component.return_value = 0.0

    ms.add_individually_powered_magnet(sim_index=10, controlled_components=["B2"],
                                       magnet_name="Q1", magnet_length=0.5)
    assert "Q1" in ms.magnets
    assert "Q1/B2" in ms.controls
    # Link connects Q1/B2 control to Q1 magnet
    assert any(lk.magnet_name == "Q1" and lk.control_name == "Q1/B2"
               for lk in ms.links.values())


# ---------------------------------------------------------------------------
# add_knob
# ---------------------------------------------------------------------------

def test_add_knob_creates_weighted_links():
    """Knob with weights creates links with scaled factors."""
    ms, mock_sc = _make_settings()
    mock_sc.lattice.get_magnet_component.return_value = 0.0

    # Set up two individually-powered magnets
    ms.add_individually_powered_magnet(sim_index=0, controlled_components=["B2"],
                                       magnet_name="Q1", magnet_length=1.0)
    ms.add_individually_powered_magnet(sim_index=1, controlled_components=["B2"],
                                       magnet_name="Q2", magnet_length=1.0)
    ms.connect_links()

    ms.add_knob("myknob", control_names=["Q1/B2", "Q2/B2"], weights=[2.0, -1.0])
    assert "myknob" in ms.controls

    # Check that knob links have correct weights
    knob_links = [lk for lk in ms.links.values() if lk.control_name == "myknob"]
    assert len(knob_links) == 2
    factors = sorted(lk.error.factor for lk in knob_links)
    assert factors == pytest.approx([-1.0, 2.0])


def test_add_knob_linearity():
    """Setting knob by k scales linked magnet contributions by k*weight."""
    ms, mock_sc = _make_settings()
    mock_sc.lattice.get_magnet_component.return_value = 0.0

    ms.add_individually_powered_magnet(sim_index=0, controlled_components=["B2"],
                                       magnet_name="Q1", magnet_length=1.0)
    ms.connect_links()

    ms.add_knob("k1", control_names=["Q1/B2"], weights=[3.0])

    # Set knob to k=2.0
    ms.set("k1", 2.0)
    # Magnet Q1's B[1] should get knob contribution: factor(3.0)*setpoint(2.0) = 6.0
    assert ms.magnets["Q1"].B[1] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# sendall
# ---------------------------------------------------------------------------

def test_sendall_updates_all_magnets():
    """sendall() calls update() on every magnet."""
    ms, _ = _make_settings()
    m1 = _add_simple_magnet(ms, name="m1", sim_index=0)
    m2 = _add_simple_magnet(ms, name="m2", sim_index=1)

    _add_simple_control(ms, name="c1")
    _add_simple_control(ms, name="c2")
    _wire_link(ms, link_name="lk1", magnet_name="m1", control_name="c1")
    _wire_link(ms, link_name="lk2", magnet_name="m2", control_name="c2")
    ms.connect_links()

    ms.controls["c1"].setpoint = 1.0
    ms.controls["c2"].setpoint = 2.0

    ms.sendall()
    assert m1.B[0] == pytest.approx(1.0)
    assert m2.B[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# REGRESSION: indentation bug in add_individually_powered_magnet
# ---------------------------------------------------------------------------

@pytest.mark.regression
def test_add_individually_powered_captures_a_offsets():
    """Verify offset detection checks ALL component types (A and B).

    Regression: an indentation bug caused the ``if not (comp in ...)`` guard
    to sit outside the inner ``for component_type`` loop, so only the B
    component was ever checked and non-zero A offsets were silently dropped.
    """
    ms, mock_sc = _make_settings()

    def fake_get(index, component_type, order, use_design=True):
        # The element has a non-zero A1 value and a non-zero B2 value
        if component_type == "A" and order == 0:
            return 0.123  # uncontrolled A component
        if component_type == "B" and order == 1:
            return 0.0  # controlled B component
        return 0.0

    mock_sc.lattice.get_magnet_component.side_effect = fake_get

    # Only control B2 -- A1 should be captured as offset_A[0]
    ms.add_individually_powered_magnet(
        sim_index=5, controlled_components=["B2"],
        magnet_name="BUG", magnet_length=1.0,
    )

    magnet = ms.magnets["BUG"]

    # With the fix, the uncontrolled A1 component IS captured in offset_A[0]
    assert magnet.offset_A[0] == pytest.approx(0.123)
