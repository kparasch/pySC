"""Tests for pySC.core.supports: Support, SupportSystem, ElementOffset."""
import pytest
import numpy as np
from unittest.mock import MagicMock

from pySC.core.supports import (
    ElementOffset,
    Support,
    SupportEndpoint,
    SupportSystem,
)
from pySC.core.transformations import at_rotation


# ---------------------------------------------------------------------------
# Reference transforms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "theta, local, world",
    [
        (0.0, np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0])),
        (np.pi / 2, np.array([1.0, 2.0, 3.0]), np.array([-1.0, 3.0, 2.0])),
        (0.37, np.array([1.0, 2.0, 3.0]), np.array([
            -np.sin(0.37) * 1.0 + np.cos(0.37) * 3.0,
             np.cos(0.37) * 1.0 + np.sin(0.37) * 3.0,
             2.0,
        ])),
    ],
)
def test_reference_pose_maps_local_to_world(theta, local, world):
    ss, _ = _make_support_system(n_elements=2, circumference=2.0)
    ss._reference_Angle[:] = theta

    _, R = ss._reference_pose(0)
    np.testing.assert_allclose(R @ local, world, atol=1e-14)
    np.testing.assert_allclose(R.T @ world, local, atol=1e-14)


def test_reference_pose_uses_element_center():
    ss, _ = _make_support_system(n_elements=20, circumference=100.0)

    p, _ = ss._reference_pose(5)

    np.testing.assert_allclose(p, np.array([27.5, 0.0, 0.0]), atol=1e-14)


# ---------------------------------------------------------------------------
# Helper: minimal mock SC for SupportSystem methods that need _parent
# ---------------------------------------------------------------------------

def _make_support_system(n_elements=20, circumference=100.0, bpm_indices=None):
    """Create a SupportSystem with a mock parent providing twiss s-positions and bpm_system."""
    ss = SupportSystem()

    mock_sc = MagicMock()
    # twiss['s'] returns an array of s-positions: evenly spaced, ending at circumference
    s_positions = np.linspace(0, circumference, n_elements + 1)  # n_elements + 1 because element 0 is at s=0
    mock_sc.lattice.twiss.__getitem__ = lambda self_dict, key: s_positions if key == 's' else None
    x_ref = s_positions.copy()
    y_ref = np.zeros_like(s_positions)
    angle_ref = np.zeros_like(s_positions)
    mock_sc.lattice.get_reference_orbit = lambda: (x_ref, y_ref, angle_ref)

    if bpm_indices is None:
        bpm_indices = []
    mock_sc.bpm_system.indices = bpm_indices

    def mock_bpm_number(index=None, name=None):
        return bpm_indices.index(index)
    mock_sc.bpm_system.bpm_number = mock_bpm_number

    # update_misalignment is called in trigger_update for non-BPM elements
    mock_sc.lattice.update_misalignment = MagicMock()

    # bpm_system fields for trigger_update
    mock_sc.bpm_system.offsets_x = np.zeros(len(bpm_indices))
    mock_sc.bpm_system.offsets_y = np.zeros(len(bpm_indices))
    mock_sc.bpm_system.rolls = np.zeros(len(bpm_indices))
    mock_sc.bpm_system.update_rot_matrices = MagicMock()

    ss._parent = mock_sc
    ss.initialize_reference_orbit()
    return ss, mock_sc


# ---------------------------------------------------------------------------
# add_element
# ---------------------------------------------------------------------------

def test_add_element():
    """add_element(index) creates an ElementOffset in L0."""
    ss, _ = _make_support_system()
    ss.add_element(5)
    assert 5 in ss.data['L0']
    assert isinstance(ss.data['L0'][5], ElementOffset)
    assert ss.data['L0'][5].index == 5
    assert ss.data['L0'][5].s == pytest.approx(27.5)


def test_add_element_duplicate_raises():
    """Adding the same index twice raises ValueError."""
    ss, _ = _make_support_system()
    ss.add_element(3)
    with pytest.raises(ValueError, match="already exists"):
        ss.add_element(3)


def test_add_element_detects_bpm():
    """Element at a BPM index gets is_bpm=True."""
    bpm_indices = [2, 7, 12]
    ss, _ = _make_support_system(bpm_indices=bpm_indices)
    ss.add_element(7)
    eo = ss.data['L0'][7]
    assert eo.is_bpm is True
    assert eo.bpm_number == 1  # index 7 is the second BPM


# ---------------------------------------------------------------------------
# add_support
# ---------------------------------------------------------------------------

def test_add_support_creates_L1():
    """add_support(start, end, level=1) creates a Support in L1."""
    ss, _ = _make_support_system()
    key = ss.add_support(2, 8, level=1, name='Girder')
    assert 'L1' in ss.data
    assert key in ss.data['L1']
    assert isinstance(ss.data['L1'][key], Support)
    assert ss.data['L1'][key].name == 'Girder'


def test_add_support_negative_index_raises():
    """Negative indices raise ValueError."""
    ss, _ = _make_support_system()
    with pytest.raises(ValueError, match="non-negative"):
        ss.add_support(-1, 5, level=1)
    with pytest.raises(ValueError, match="non-negative"):
        ss.add_support(5, -1, level=1)


def test_add_support_calculates_length():
    """Support length = (end_s - start_s) mod circumference."""
    circumference = 100.0
    ss, _ = _make_support_system(n_elements=20, circumference=circumference)
    # Element centers are at s = 2.5, 7.5, 12.5, ...
    # Element 2 center at s=12.5, element 8 center at s=42.5
    key = ss.add_support(2, 8, level=1)
    support = ss.data['L1'][key]
    assert support.start.s == pytest.approx(12.5)
    assert support.end.s == pytest.approx(42.5)
    assert support.length == pytest.approx(30.0)  # 42.5 - 12.5


# ---------------------------------------------------------------------------
# resolve_graph
# ---------------------------------------------------------------------------

def test_resolve_graph_assigns_supported_by():
    """After resolve, elements within a support have supported_by set."""
    ss, _ = _make_support_system(n_elements=20)
    # Add elements
    for i in [3, 5, 7]:
        ss.add_element(i)
    # Add support covering indices 2..8
    supp_key = ss.add_support(2, 8, level=1)
    ss.resolve_graph()

    for i in [3, 5, 7]:
        assert ss.data['L0'][i].supported_by == ('L1', supp_key)


def test_resolve_graph_populates_supports_elements():
    """After resolve, supports list their supported elements."""
    ss, _ = _make_support_system(n_elements=20)
    for i in [3, 5, 7, 15]:
        ss.add_element(i)
    supp_key = ss.add_support(2, 8, level=1)
    ss.resolve_graph()

    support = ss.data['L1'][supp_key]
    supported_indices = [idx for _, idx in support.supports_elements]
    # Elements 3, 5, 7 are inside [2, 8], element 15 is outside
    assert 3 in supported_indices
    assert 5 in supported_indices
    assert 7 in supported_indices
    assert 15 not in supported_indices


def test_resolve_graph_wrapping_support():
    """Support crossing s=0 correctly contains elements near start/end of ring."""
    ss, _ = _make_support_system(n_elements=20)
    # Add elements near start and end of ring
    for i in [0, 1, 18, 19]:
        ss.add_element(i)
    # Wrapping support: start > end means it wraps through s=0
    supp_key = ss.add_support(18, 1, level=1)
    ss.resolve_graph()

    support = ss.data['L1'][supp_key]
    supported_indices = [idx for _, idx in support.supports_elements]
    # Elements 18, 19, 0, 1 should all be inside the wrapping support
    assert 18 in supported_indices
    assert 19 in supported_indices
    assert 0 in supported_indices
    assert 1 in supported_indices


# ---------------------------------------------------------------------------
# get_total_offset
# ---------------------------------------------------------------------------

def test_get_total_offset_unsupported():
    """Element with no support returns its own 3D offset."""
    ss, _ = _make_support_system(n_elements=20)
    ss.add_element(5)
    ss.data['L0'][5].dx = 0.001
    ss.data['L0'][5].dy = 0.002
    ss.data['L0'][5].ds = 0.003

    offset = ss.get_total_offset(5)
    np.testing.assert_allclose(offset, np.array([0.001, 0.002, 0.003]), atol=1e-14)


def test_get_total_offset_one_level_translation():
    """Element on a translated support gets a 3D parent offset plus its own offset."""
    ss, _ = _make_support_system(n_elements=20, circumference=100.0)
    ss.add_element(5)  # center s=27.5
    supp_key = ss.add_support(2, 8, level=1)  # center s: 12.5 to 42.5, length=30
    ss.resolve_graph()

    support = ss.data['L1'][supp_key]
    support.start.dx = 0.010
    support.start.dy = 0.020
    support.start.ds = 0.030
    support.end.dx = 0.010
    support.end.dy = 0.020
    support.end.ds = 0.030

    ss.data['L0'][5].dx = 0.001
    ss.data['L0'][5].dy = 0.002
    ss.data['L0'][5].ds = 0.003

    offset = ss.get_total_offset(5)
    np.testing.assert_allclose(offset, np.array([0.011, 0.022, 0.033]), atol=1e-14)


def test_support_roll_rotates_child_element_offset():
    """Child element offsets are composed through the support rotation matrix."""
    ss, _ = _make_support_system(n_elements=20, circumference=100.0)
    ss.add_element(5)
    supp_key = ss.add_support(2, 8, level=1)
    ss.resolve_graph()

    ss.data['L1'][supp_key].roll = np.pi / 2
    ss.data['L0'][5].dx = 1.0

    offset = ss.get_total_offset(5)
    np.testing.assert_allclose(offset, np.array([0.0, 1.0, 0.0]), atol=1e-14)


def test_support_endpoint_ds_propagates_to_element_longitudinal_offset():
    """Endpoint ds contributes to the resolved element longitudinal component."""
    ss, _ = _make_support_system(n_elements=20, circumference=100.0)
    ss.add_element(5)
    supp_key = ss.add_support(2, 8, level=1)
    ss.resolve_graph()

    support = ss.data['L1'][supp_key]
    support.start.ds = 0.1
    support.end.ds = 0.1

    offset = ss.get_total_offset(5)
    np.testing.assert_allclose(offset, np.array([0.0, 0.0, 0.1]), atol=1e-14)


# ---------------------------------------------------------------------------
# get_support_offset linear interpolation
# ---------------------------------------------------------------------------

def test_get_support_offset_linear_interpolation():
    """Midpoint of support gets the mean of endpoint offsets in 3D."""
    ss, _ = _make_support_system(n_elements=20, circumference=100.0)
    supp_key = ss.add_support(4, 8, level=1)
    support = ss.data['L1'][supp_key]
    support.start.dx = 0.0
    support.start.dy = 0.0
    support.start.ds = 0.0
    support.end.dx = 1.0
    support.end.dy = 2.0
    support.end.ds = 3.0

    midpoint_s = 0.5 * (support.start.s + support.end.s)
    offset = ss.get_support_offset(midpoint_s, ('L1', supp_key))
    np.testing.assert_allclose(offset, np.array([0.5, 1.0, 1.5]), atol=1e-14)


def test_get_support_offset_wrapping():
    """Support crossing s=0 interpolates correctly for elements on both sides."""
    circumference = 100.0
    ss, _ = _make_support_system(n_elements=20, circumference=circumference)
    # Support from index 18 (center s=92.5) to index 2 (center s=12.5), wrapping
    # length = (12.5 - 92.5) % 100 = 20
    supp_key = ss.add_support(18, 2, level=1)
    support = ss.data['L1'][supp_key]
    support.start.dx = 0.0
    support.start.dy = 0.0
    support.start.ds = 0.0
    support.end.dx = 1.0
    support.end.dy = 2.0
    support.end.ds = 3.0

    # Element 0 center is s=2.5, halfway along the wrapped support.
    offset_at_element_0 = ss.get_support_offset(ss._element_center_s(0), ('L1', supp_key))
    np.testing.assert_allclose(offset_at_element_0, np.array([0.5, 1.0, 1.5]), atol=1e-14)


def test_non_rigid_support_keeps_endpoint_distance_change():
    """Non-rigid supports keep the endpoint positions created by endpoint offsets."""
    ss, _ = _make_support_system(n_elements=20, circumference=100.0)
    supp_key = ss.add_support(2, 8, level=1)
    support = ss.data['L1'][supp_key]
    support.end.ds = 1.0

    start_offset = ss.get_total_offset(supp_key, level='L1', endpoint='start')
    end_offset = ss.get_total_offset(supp_key, level='L1', endpoint='end')
    start_ref, R_start = ss._reference_pose(support.start.index)
    end_ref, R_end = ss._reference_pose(support.end.index)
    start = start_ref + R_start @ start_offset
    end = end_ref + R_end @ end_offset

    assert np.linalg.norm(end - start) == pytest.approx(31.0)


def test_rigid_support_preserves_nominal_endpoint_distance():
    """Rigid supports rescale misaligned endpoints to preserve nominal chord length."""
    ss, _ = _make_support_system(n_elements=20, circumference=100.0)
    supp_key = ss.add_support(2, 8, level=1)
    support = ss.data['L1'][supp_key]
    support.rigid = True
    support.end.ds = 1.0

    start_offset = ss.get_total_offset(supp_key, level='L1', endpoint='start')
    end_offset = ss.get_total_offset(supp_key, level='L1', endpoint='end')
    start_ref, R_start = ss._reference_pose(support.start.index)
    end_ref, R_end = ss._reference_pose(support.end.index)
    start = start_ref + R_start @ start_offset
    end = end_ref + R_end @ end_offset

    assert np.linalg.norm(end - start) == pytest.approx(30.0)


def test_mixed_parent_support_uses_resolved_endpoint_positions():
    """A support can be defined by endpoints resolved from different parents."""
    ss, _ = _make_support_system(n_elements=20, circumference=100.0)
    child_key = ss.add_support(5, 15, level=1)
    left_parent_key = ss.add_support(0, 9, level=2)
    right_parent_key = ss.add_support(10, 19, level=2)
    ss.resolve_graph()

    assert ('L1', child_key) in ss.data['L2'][left_parent_key].supports_elements
    assert ('L1', child_key) in ss.data['L2'][right_parent_key].supports_elements

    ss.data['L2'][left_parent_key].start.dx = 1.0
    ss.data['L2'][left_parent_key].end.dx = 1.0
    ss.data['L2'][right_parent_key].start.dx = 3.0
    ss.data['L2'][right_parent_key].end.dx = 3.0

    child = ss.data['L1'][child_key]
    offset = ss.get_support_offset(0.5 * (child.start.s + child.end.s), ('L1', child_key))
    np.testing.assert_allclose(offset, np.array([2.0, 0.0, 0.0]), atol=1e-14)


# ---------------------------------------------------------------------------
# get_total_rotation
# ---------------------------------------------------------------------------

def test_get_total_rotation_with_support():
    """Support and element rotations compose as matrices, not scalar sums."""
    ss, _ = _make_support_system(n_elements=20, circumference=100.0)
    ss.add_element(5)  # center s=27.5
    supp_key = ss.add_support(2, 8, level=1)  # center s: 12.5 to 42.5
    ss.resolve_graph()

    # Set element rotation
    ss.data['L0'][5].roll = 0.01
    ss.data['L0'][5].yaw = 0.02
    ss.data['L0'][5].pitch = 0.03

    ss.data['L1'][supp_key].roll = 0.1

    resolved = ss.get_total_rotation(5).as_matrix()
    expected = (
        at_rotation(roll=0.1).as_matrix()
        @ at_rotation(pitch=0.03, yaw=0.02, roll=0.01).as_matrix()
    )
    np.testing.assert_allclose(resolved, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# Integration tests using the sc fixture (require AT)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_set_offset_triggers_update(sc):
    """set_offset(index, dx=0.001) propagates to lattice misalignment."""
    # Register elements and a support
    ss = sc.support_system
    quad_idx = [i for i, e in enumerate(sc.lattice.design)
                if hasattr(e, 'FamName') and 'Q' in e.FamName]
    if len(quad_idx) < 2:
        pytest.skip("Need at least 2 quads")

    idx = quad_idx[0]
    ss.add_element(idx)
    ss.set_offset(idx, dx=0.001, dy=0.002)

    eo = ss.data['L0'][idx]
    assert eo.dx == pytest.approx(0.001)
    assert eo.dy == pytest.approx(0.002)


@pytest.mark.slow
def test_trigger_update_propagates_to_bpm(sc):
    """Setting offset on BPM element updates bpm_system.offsets_x/y."""
    ss = sc.support_system
    bpm_idx = sc.bpm_system.indices[0]
    ss.add_element(bpm_idx)

    offset_val = 0.0005
    ss.set_offset(bpm_idx, dx=offset_val, dy=offset_val * 2)

    # The trigger_update for a BPM writes directly to bpm_system.offsets_x/y
    assert sc.bpm_system.offsets_x[0] == pytest.approx(offset_val)
    assert sc.bpm_system.offsets_y[0] == pytest.approx(offset_val * 2)

    # Clean up
    ss.set_offset(bpm_idx, dx=0.0, dy=0.0)


@pytest.mark.slow
def test_trigger_update_propagates_to_magnet(sc):
    """Setting offset on magnet element updates lattice element T1/T2."""
    import at
    ss = sc.support_system
    quad_indices = [i for i, e in enumerate(sc.lattice.design) if isinstance(e, at.Quadrupole)]
    idx = quad_indices[0]
    ss.add_element(idx)

    # Get baseline T1
    elem = sc.lattice.ring[idx]
    t1_before = elem.T1.copy() if hasattr(elem, 'T1') else np.zeros(6)

    ss.set_offset(idx, dx=0.001)

    # After update, T1 should differ (misalignment applied)
    t1_after = elem.T1.copy()
    assert not np.allclose(t1_after, t1_before), \
        "T1 should change after applying dx offset"

    # Clean up
    ss.set_offset(idx, dx=0.0)


@pytest.mark.slow
def test_update_all(sc):
    """update_all() propagates offsets for every L0 element."""
    import at
    ss = sc.support_system

    # Register a few elements
    quad_indices = [i for i, e in enumerate(sc.lattice.design) if isinstance(e, at.Quadrupole)]
    bpm_idx = sc.bpm_system.indices[0]

    for idx in quad_indices[:3]:
        if idx not in ss.data['L0']:
            ss.add_element(idx)
    if bpm_idx not in ss.data['L0']:
        ss.add_element(bpm_idx)

    # Set offsets on all registered elements
    for idx in ss.data['L0'].keys():
        ss.data['L0'][idx].dx = 0.001

    # Capture T1 state before update_all
    t1_before = {}
    for idx in ss.data['L0'].keys():
        eo = ss.data['L0'][idx]
        if not eo.is_bpm:
            elem = sc.lattice.ring[idx]
            t1_before[idx] = elem.T1.copy() if hasattr(elem, 'T1') else np.zeros(6)

    ss.update_all()

    # At least some non-BPM elements should have changed T1
    changed = 0
    for idx, t1_old in t1_before.items():
        elem = sc.lattice.ring[idx]
        if hasattr(elem, 'T1') and not np.allclose(elem.T1, t1_old):
            changed += 1
    assert changed > 0, "update_all() should propagate offsets to lattice elements"

    # Clean up
    for idx in list(ss.data['L0'].keys()):
        ss.data['L0'][idx].dx = 0.0
    ss.update_all()
