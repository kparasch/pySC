"""Tests for pySC.core.supports: Support, SupportSystem, ElementOffset."""
import pytest
import numpy as np
from unittest.mock import MagicMock, PropertyMock, patch

from pySC.core.supports import (
    ElementOffset,
    Support,
    SupportEndpoint,
    SupportSystem,
)


# ---------------------------------------------------------------------------
# Support yaw / pitch properties
# ---------------------------------------------------------------------------

def test_support_yaw_pitch_zero_length():
    """Support with length=0 returns yaw=0, pitch=0."""
    s = Support(
        start=SupportEndpoint(index=0, dx=0.1, dy=0.2, s=0.0),
        end=SupportEndpoint(index=5, dx=0.3, dy=0.5, s=0.0),
        length=0.0,
    )
    assert s.yaw == 0.0
    assert s.pitch == 0.0


def test_support_yaw_pitch_calculation():
    """Yaw = (end.dx - start.dx)/length, Pitch = (end.dy - start.dy)/length."""
    s = Support(
        start=SupportEndpoint(index=0, dx=0.1, dy=0.2, s=0.0),
        end=SupportEndpoint(index=10, dx=0.5, dy=0.8, s=2.0),
        length=2.0,
    )
    assert s.yaw == pytest.approx((0.5 - 0.1) / 2.0)
    assert s.pitch == pytest.approx((0.8 - 0.2) / 2.0)


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
    # Elements are at s = 0, 5, 10, ..., 100
    # Element 2 at s=10, element 8 at s=40
    key = ss.add_support(2, 8, level=1)
    support = ss.data['L1'][key]
    assert support.length == pytest.approx(30.0)  # 40 - 10


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
    """Element with no support returns its own (dx, dy)."""
    ss, _ = _make_support_system(n_elements=20)
    ss.add_element(5)
    ss.data['L0'][5].dx = 0.001
    ss.data['L0'][5].dy = 0.002

    dx, dy = ss.get_total_offset(5)
    assert dx == pytest.approx(0.001)
    assert dy == pytest.approx(0.002)


def test_get_total_offset_one_level():
    """Element on a girder: total = own offset + interpolated girder offset."""
    ss, _ = _make_support_system(n_elements=20, circumference=100.0)
    # Elements at s = 0, 5, 10, ..., 100
    ss.add_element(5)  # s=25
    supp_key = ss.add_support(2, 8, level=1)  # s: 10 to 40, length=30
    ss.resolve_graph()

    # Set girder endpoint offsets
    ss.data['L1'][supp_key].start.dx = 0.010
    ss.data['L1'][supp_key].start.dy = 0.020
    ss.data['L1'][supp_key].end.dx = 0.040
    ss.data['L1'][supp_key].end.dy = 0.080

    # Element at s=25, support from s=10 to s=40 (length=30)
    # Interpolated girder dx: (0.040 - 0.010)/(40-10) * (25-10) + 0.010 = 0.030/30 * 15 + 0.010 = 0.025
    # Interpolated girder dy: (0.080 - 0.020)/(40-10) * (25-10) + 0.020 = 0.060/30 * 15 + 0.020 = 0.050

    # Element own offset
    ss.data['L0'][5].dx = 0.001
    ss.data['L0'][5].dy = 0.002

    dx, dy = ss.get_total_offset(5)
    assert dx == pytest.approx(0.001 + 0.025)
    assert dy == pytest.approx(0.002 + 0.050)


# ---------------------------------------------------------------------------
# get_support_offset linear interpolation
# ---------------------------------------------------------------------------

def test_get_support_offset_linear_interpolation():
    """Element at midpoint of support gets the mean of endpoint offsets."""
    ss, _ = _make_support_system(n_elements=20, circumference=100.0)
    # Support from index 4 (s=20) to index 8 (s=40), length=20
    supp_key = ss.add_support(4, 8, level=1)
    support = ss.data['L1'][supp_key]
    support.start.dx = 0.0
    support.start.dy = 0.0
    support.end.dx = 1.0
    support.end.dy = 2.0

    # Midpoint s = 30
    offset = ss.get_support_offset(30.0, ('L1', supp_key))
    assert offset[0] == pytest.approx(0.5)
    assert offset[1] == pytest.approx(1.0)


def test_get_support_offset_wrapping():
    """Support crossing s=0 interpolates correctly for elements on both sides."""
    circumference = 100.0
    ss, _ = _make_support_system(n_elements=20, circumference=circumference)
    # Support from index 18 (s=90) to index 2 (s=10), wrapping
    # length = (10 - 90) % 100 = 20
    supp_key = ss.add_support(18, 2, level=1)
    support = ss.data['L1'][supp_key]
    support.start.dx = 0.0
    support.start.dy = 0.0
    support.end.dx = 1.0
    support.end.dy = 2.0

    # Element at s=0 (near the wrap point)
    # s=0 < s1=90, so corr_s = circumference = 100
    # corr_s2 = circumference = 100 (because start.index > end.index)
    # dx = (1.0 - 0.0)/(10 - 90 + 100) * (0 - 90 + 100) + 0.0 = 1.0/20 * 10 = 0.5
    offset_at_0 = ss.get_support_offset(0.0, ('L1', supp_key))
    assert offset_at_0[0] == pytest.approx(0.5)
    assert offset_at_0[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# get_total_rotation
# ---------------------------------------------------------------------------

def test_get_total_rotation_with_support():
    """roll/yaw/pitch from support adds to element's own rotation."""
    ss, _ = _make_support_system(n_elements=20, circumference=100.0)
    ss.add_element(5)  # s=25
    supp_key = ss.add_support(2, 8, level=1)  # s: 10 to 40
    ss.resolve_graph()

    # Set element rotation
    ss.data['L0'][5].roll = 0.01
    ss.data['L0'][5].yaw = 0.02
    ss.data['L0'][5].pitch = 0.03

    # Set support roll (yaw/pitch come from endpoint dx/dy differences)
    ss.data['L1'][supp_key].roll = 0.1
    ss.data['L1'][supp_key].start.dx = 0.0
    ss.data['L1'][supp_key].end.dx = 0.6   # yaw = 0.6/30 = 0.02
    ss.data['L1'][supp_key].start.dy = 0.0
    ss.data['L1'][supp_key].end.dy = 0.9   # pitch = 0.9/30 = 0.03

    roll, pitch, yaw = ss.get_total_rotation(5)
    assert roll == pytest.approx(0.01 + 0.1)     # element + support roll
    assert yaw == pytest.approx(0.02 + 0.02)     # element + support yaw
    assert pitch == pytest.approx(0.03 + 0.03)   # element + support pitch


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
