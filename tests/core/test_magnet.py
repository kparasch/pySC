"""Tests for pySC.core.magnet: Magnet, ControlMagnetLink."""
import pytest
from unittest.mock import MagicMock
from pySC.core.magnet import Magnet, ControlMagnetLink
from pySC.core.control import Control, LinearConv


# ---------------------------------------------------------------------------
# Magnet initialisation
# ---------------------------------------------------------------------------

def test_magnet_initialization_defaults():
    """Magnet with max_order=2 allocates A, B, offset_A, offset_B of length 3."""
    m = Magnet(max_order=2, sim_index=0)
    assert len(m.A) == 3
    assert len(m.B) == 3
    assert len(m.offset_A) == 3
    assert len(m.offset_B) == 3
    assert m.A == [0.0, 0.0, 0.0]


def test_magnet_name_defaults_to_sim_index():
    """When name is not provided but sim_index=5, name is set to 5."""
    m = Magnet(max_order=1, sim_index=5)
    assert m.name == 5


# ---------------------------------------------------------------------------
# ControlMagnetLink.value
# ---------------------------------------------------------------------------

def test_control_magnet_link_value():
    """Link.value(setpoint) applies the error transform (factor*x + offset)."""
    link = ControlMagnetLink(
        link_name="lk",
        magnet_name="m1",
        control_name="c1",
        component="B",
        order=1,
        error=LinearConv(factor=2.0, offset=0.5),
    )
    assert link.value(3.0) == pytest.approx(6.5)  # 2*3 + 0.5


# ---------------------------------------------------------------------------
# Magnet.update
# ---------------------------------------------------------------------------

def _make_magnet_with_parent(max_order=1, length=1.0):
    """Helper: create a Magnet wired to a mock MagnetSettings/SC so update() works."""
    m = Magnet(max_order=max_order, sim_index=0, length=length)
    mock_lattice = MagicMock()
    mock_sc = MagicMock()
    mock_sc.lattice = mock_lattice

    mock_parent = MagicMock()
    mock_parent._parent = mock_sc
    mock_parent.controls = {}

    m._parent = mock_parent
    return m, mock_parent


def test_magnet_update_resets_to_offsets():
    """update() starts from offset_A/B before adding link contributions."""
    m, parent = _make_magnet_with_parent(max_order=1, length=1.0)
    m.offset_B[0] = 0.1

    # Add a link that contributes to B[0]
    ctrl = Control(name="c1", setpoint=5.0)
    parent.controls["c1"] = ctrl
    link = ControlMagnetLink(
        link_name="lk1", magnet_name=0, control_name="c1",
        component="B", order=1, error=LinearConv(factor=1.0, offset=0.0),
    )
    m._links = [link]

    m.update()
    # B[0] should be offset (0.1) + setpoint contribution (5.0)
    assert m.B[0] == pytest.approx(5.1)

    # Call update again -- should reset, not accumulate
    m.update()
    assert m.B[0] == pytest.approx(5.1)


def test_magnet_update_integrated_strength():
    """Link with is_integrated=True divides link value by magnet length."""
    m, parent = _make_magnet_with_parent(max_order=1, length=2.0)

    ctrl = Control(name="c1", setpoint=6.0)
    parent.controls["c1"] = ctrl
    link = ControlMagnetLink(
        link_name="lk1", magnet_name=0, control_name="c1",
        component="B", order=1, is_integrated=True,
    )
    m._links = [link]

    m.update()
    # 6.0 / 2.0 = 3.0
    assert m.B[0] == pytest.approx(3.0)


def test_magnet_update_no_length_raises():
    """Integrated link on a magnet with length=None raises AssertionError."""
    m, parent = _make_magnet_with_parent(max_order=1, length=None)
    # Explicitly set length to None after construction (model_validator may set it)
    m.length = None

    ctrl = Control(name="c1", setpoint=1.0)
    parent.controls["c1"] = ctrl
    link = ControlMagnetLink(
        link_name="lk1", magnet_name=0, control_name="c1",
        component="B", order=1, is_integrated=True,
    )
    m._links = [link]

    with pytest.raises(AssertionError, match="magnet length not specified"):
        m.update()


def test_shifted_magnet_update():
    """Shifted quadrupole feeds down"""

    design_arc_length = 0.25
    design_shift = 0.01
    initial_k1 = 5.0
    initial_bending_angle = design_shift * initial_k1 * design_arc_length
    initial_b0 = 0.02

    m, parent = _make_magnet_with_parent(max_order=1, length=design_arc_length)
    m.is_shifted = True
    m.bending_length = design_arc_length
    m.design_shift = design_shift
    m.design_k1 = initial_k1
    m.offset_B[0] = initial_b0

    element = MagicMock()
    element.Length = design_arc_length
    element.BendingAngle = initial_bending_angle
    element.EntranceAngle = initial_bending_angle / 2
    element.ExitAngle = initial_bending_angle / 2

    support_system = MagicMock()
    support_system.get_total_offset.return_value = (0.0, 0.0)
    support_system.get_total_rotation.return_value = (0.0, 0.0, 0.0)
    support_system.data = {'L0': {0: MagicMock(dz=0.0)}}
    parent._parent.support_system = support_system
    parent._parent.lattice.ring = {0: element}
    parent._parent.lattice.design = {0: element}

    ctrl = Control(name="c1", setpoint=initial_k1 * design_arc_length)
    parent.controls["c1"] = ctrl
    link = ControlMagnetLink(link_name="lk1", magnet_name=0, control_name="c1", component="B", order=2, is_integrated=True)
    m._links = [link]

    m.update()

    expected_k1 = initial_k1
    expected_b0 = initial_b0
    expected_angle = initial_bending_angle
    assert m.B[1] == pytest.approx(expected_k1)
    assert m.B[0] == pytest.approx(expected_b0)
    assert element.Length == pytest.approx(design_arc_length)
    assert element.BendingAngle == pytest.approx(expected_angle)
    assert element.EntranceAngle == pytest.approx(expected_angle/2)
    assert element.ExitAngle == pytest.approx(expected_angle/2)
    assert m.length == pytest.approx(design_arc_length)
    assert m.bending_length == pytest.approx(design_arc_length)

    ctrl.setpoint = 5.2 * design_arc_length
    m.update()

    expected_k1 = 5.2
    expected_b0 = initial_b0 + design_shift * (expected_k1 - initial_k1)
    assert m.B[1] == pytest.approx(expected_k1)
    assert m.B[0] == pytest.approx(expected_b0)
    assert element.Length == pytest.approx(design_arc_length)
    assert element.BendingAngle == pytest.approx(initial_bending_angle)
    assert element.EntranceAngle == pytest.approx(initial_bending_angle/2)
    assert element.ExitAngle == pytest.approx(initial_bending_angle/2)
    assert m.length == pytest.approx(design_arc_length)
    assert m.bending_length == pytest.approx(design_arc_length)

    support_system.get_total_offset.return_value = (0.001, 0.0)
    m.update()

    total_shift = design_shift + 0.001
    expected_b0 = initial_b0 + total_shift * expected_k1 - design_shift * initial_k1
    assert m.B[1] == pytest.approx(expected_k1)
    assert m.B[0] == pytest.approx(expected_b0)
    assert element.Length == pytest.approx(design_arc_length)
    assert element.BendingAngle == pytest.approx(initial_bending_angle)
    assert element.EntranceAngle == pytest.approx(initial_bending_angle/2)
    assert element.ExitAngle == pytest.approx(initial_bending_angle/2)
    assert m.length == pytest.approx(design_arc_length)
    assert m.bending_length == pytest.approx(design_arc_length)
