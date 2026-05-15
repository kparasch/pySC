"""Tests for pySC.utils.sc_tools: rotation matrices and coordinate transformations."""
import numpy as np
import pytest
from types import SimpleNamespace

from pySC.utils.sc_tools import rotation, update_transformation


class TestRotation:

    def test_rotation_identity(self):
        """rotation([0,0,0]) should produce the 3x3 identity matrix."""
        R = rotation([0, 0, 0])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_rotation_single_axis_z(self):
        """Pure roll (z-axis rotation) matches the standard 2D rotation in the xy-plane."""
        theta = np.pi / 6
        R = rotation([0, 0, theta])
        expected = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1],
        ])
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_rotation_single_axis_y(self):
        """Pure yaw (y-axis rotation) matches the standard Ry matrix."""
        theta = np.pi / 4
        R = rotation([0, theta, 0])
        expected = np.array([
            [ np.cos(theta), 0, np.sin(theta)],
            [ 0,             1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ])
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_rotation_single_axis_x(self):
        """Pure pitch (x-axis rotation) matches the standard Rx matrix."""
        theta = np.pi / 3
        R = rotation([theta, 0, 0])
        expected = np.array([
            [1, 0,              0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)],
        ])
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_rotation_composition(self):
        """Two sequential rotations applied via matrix multiplication equal the combined rotation.

        The rotation function implements extrinsic ZYX (Roll, Yaw, Pitch), so
        rotation([ax1+ax2, ay1+ay2, az1+az2]) is NOT in general the same as
        rotation([ax1, ay1, az1]) @ rotation([ax2, ay2, az2]).
        However, sequential single-axis rotations about the SAME axis compose additively.
        """
        theta1 = 0.1
        theta2 = 0.2
        # Z-axis rotations compose additively
        R_combined = rotation([0, 0, theta1 + theta2])
        R_sequential = rotation([0, 0, theta1]) @ rotation([0, 0, theta2])
        np.testing.assert_allclose(R_combined, R_sequential, atol=1e-14)

        # Y-axis rotations compose additively
        R_combined = rotation([0, theta1 + theta2, 0])
        R_sequential = rotation([0, theta1, 0]) @ rotation([0, theta2, 0])
        np.testing.assert_allclose(R_combined, R_sequential, atol=1e-14)


def _make_element(length=0.0, bending_angle=0.0):
    """Create a minimal mock element with the attributes update_transformation sets."""
    elem = SimpleNamespace(
        Length=length,
        BendingAngle=bending_angle,
        R1=None, R2=None, T1=None, T2=None,
    )
    return elem


class TestUpdateTransformation:

    def test_update_transformation_zero_offsets(self):
        """With all offsets and rolls zero, T1/T2 should be near-zero, R1/R2 near-identity."""
        elem = _make_element(length=1.0)
        result = update_transformation(elem, dx=0, dy=0, dz=0, roll=0, yaw=0, pitch=0)
        np.testing.assert_allclose(result.T1, np.zeros(6), atol=1e-14)
        np.testing.assert_allclose(result.T2, np.zeros(6), atol=1e-14)
        np.testing.assert_allclose(result.R1, np.eye(6), atol=1e-14)
        np.testing.assert_allclose(result.R2, np.eye(6), atol=1e-14)

    def test_update_transformation_dx_only(self):
        """Horizontal offset only should modify T1[0] and T2[0]."""
        dx_val = 0.001  # 1 mm
        elem = _make_element(length=1.0)
        result = update_transformation(elem, dx=dx_val, dy=0, dz=0, roll=0, yaw=0, pitch=0)

        # T1[0] should have the horizontal translation component
        assert abs(result.T1[0]) > 1e-6, "T1[0] should be non-zero for dx offset"
        # T2[0] should also reflect the offset
        assert abs(result.T2[0]) > 1e-6, "T2[0] should be non-zero for dx offset"
        # Vertical components should remain zero
        np.testing.assert_allclose(result.T1[2], 0.0, atol=1e-14)
        np.testing.assert_allclose(result.T2[2], 0.0, atol=1e-14)
        # R1/R2 should remain identity since there are no rotations
        np.testing.assert_allclose(result.R1, np.eye(6), atol=1e-14)
        np.testing.assert_allclose(result.R2, np.eye(6), atol=1e-14)

    def test_update_transformation_roll_only(self):
        """Roll-only should modify R1/R2 rotation blocks while keeping T1/T2 near-zero."""
        roll_val = 0.01  # 10 mrad
        elem = _make_element(length=1.0)
        result = update_transformation(elem, dx=0, dy=0, dz=0, roll=roll_val, yaw=0, pitch=0)

        # R1 and R2 should differ from identity
        assert not np.allclose(result.R1, np.eye(6), atol=1e-6), \
            "R1 should differ from identity for non-zero roll"
        assert not np.allclose(result.R2, np.eye(6), atol=1e-6), \
            "R2 should differ from identity for non-zero roll"

        # T1 and T2 should remain near zero (no offsets applied)
        np.testing.assert_allclose(result.T1, np.zeros(6), atol=1e-14)
        np.testing.assert_allclose(result.T2, np.zeros(6), atol=1e-14)

    def test_update_transformation_with_dipole(self):
        """Dipole with BendingAngle should apply exit angle corrections to T2/R2."""
        angle = 0.05  # 50 mrad bending angle
        length = 2.0
        elem = _make_element(length=length, bending_angle=angle)

        # Apply a small horizontal offset to a dipole
        dx_val = 0.001
        result = update_transformation(elem, dx=dx_val, dy=0, dz=0, roll=0, yaw=0, pitch=0)

        # For comparison, compute the same with a drift (no bending)
        elem_drift = _make_element(length=length, bending_angle=0.0)
        result_drift = update_transformation(elem_drift, dx=dx_val, dy=0, dz=0, roll=0, yaw=0, pitch=0)

        # T2 should differ between dipole and drift due to exit angle corrections
        assert not np.allclose(result.T2, result_drift.T2, atol=1e-8), \
            "Dipole T2 should differ from drift T2 due to BendingAngle"
        # R2 should also differ
        assert not np.allclose(result.R2, result_drift.R2, atol=1e-8), \
            "Dipole R2 should differ from drift R2 due to BendingAngle"

        # T1 and R1 should be the same (entrance is unaffected by BendingAngle)
        np.testing.assert_allclose(result.T1, result_drift.T1, atol=1e-14)
        np.testing.assert_allclose(result.R1, result_drift.R1, atol=1e-14)

    def test_update_transformation_modifies_element_in_place(self):
        """update_transformation returns the same element object (mutated in place)."""
        elem = _make_element(length=1.0)
        result = update_transformation(elem, dx=0.001, dy=0, dz=0, roll=0, yaw=0, pitch=0)
        assert result is elem
