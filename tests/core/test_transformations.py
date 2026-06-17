"""Tests for pySC.core.transformations."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pySC.core.transformations import (
    at_angles_from_rotation,
    at_rotation,
    at_rotation_matrix,
    update_at_transformation,
    xsuite_angles_from_rotation,
)


class TestRotation:
    def test_rotation_identity(self):
        np.testing.assert_allclose(at_rotation_matrix(), np.eye(3), atol=1e-15)

    def test_rotation_single_axis_z(self):
        theta = np.pi / 6
        R = at_rotation_matrix(roll=theta)
        expected = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1],
        ])
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_rotation_single_axis_y(self):
        theta = np.pi / 4
        R = at_rotation_matrix(yaw=theta)
        expected = np.array([
            [ np.cos(theta), 0, np.sin(theta)],
            [ 0,             1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ])
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_rotation_single_axis_x(self):
        theta = np.pi / 3
        R = at_rotation_matrix(pitch=theta)
        expected = np.array([
            [1, 0,              0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)],
        ])
        np.testing.assert_allclose(R, expected, atol=1e-14)

    def test_rotation_composition(self):
        theta1 = 0.1
        theta2 = 0.2

        R_combined = at_rotation_matrix(roll=theta1 + theta2)
        R_sequential = at_rotation_matrix(roll=theta1) @ at_rotation_matrix(roll=theta2)
        np.testing.assert_allclose(R_combined, R_sequential, atol=1e-14)

        R_combined = at_rotation_matrix(yaw=theta1 + theta2)
        R_sequential = at_rotation_matrix(yaw=theta1) @ at_rotation_matrix(yaw=theta2)
        np.testing.assert_allclose(R_combined, R_sequential, atol=1e-14)

    def test_rotation_matches_scipy_convention(self):
        pitch = 0.31
        yaw = -0.22
        roll = 0.17
        expected = Rotation.from_euler('zyx', [roll, yaw, pitch]).as_matrix()
        np.testing.assert_allclose(at_rotation_matrix(pitch=pitch, yaw=yaw, roll=roll), expected, atol=1e-14)

    def test_rotation_angle_round_trip(self):
        pitch = 0.03
        yaw = -0.02
        roll = 0.01
        out_roll, out_pitch, out_yaw = at_angles_from_rotation(
            at_rotation_matrix(pitch=pitch, yaw=yaw, roll=roll)
        )
        assert out_roll == pytest.approx(roll)
        assert out_pitch == pytest.approx(pitch)
        assert out_yaw == pytest.approx(yaw)

    def test_xsuite_single_axis_angle_conversion(self):
        theta = 0.03

        rot_s, rot_x, rot_y = xsuite_angles_from_rotation(at_rotation(roll=theta))
        assert rot_s == pytest.approx(theta)
        assert rot_x == pytest.approx(0.0)
        assert rot_y == pytest.approx(0.0)

        rot_s, rot_x, rot_y = xsuite_angles_from_rotation(at_rotation(pitch=theta))
        assert rot_s == pytest.approx(0.0)
        assert rot_x == pytest.approx(-theta)
        assert rot_y == pytest.approx(0.0)

        rot_s, rot_x, rot_y = xsuite_angles_from_rotation(at_rotation(yaw=theta))
        assert rot_s == pytest.approx(0.0)
        assert rot_x == pytest.approx(0.0)
        assert rot_y == pytest.approx(theta)

    def test_xsuite_angle_conversion_reconstructs_xtrack_matrix(self):
        rot = at_rotation(pitch=0.11, yaw=-0.07, roll=0.05)
        rot_s_rad_no_frame, rot_x_rad, rot_y_rad = xsuite_angles_from_rotation(rot)

        s_phi, c_phi = np.sin(rot_x_rad), np.cos(rot_x_rad)
        s_theta, c_theta = np.sin(rot_y_rad), np.cos(rot_y_rad)
        s_psi, c_psi = np.sin(rot_s_rad_no_frame), np.cos(rot_s_rad_no_frame)
        xtrack_matrix = np.array([
            [-s_phi * s_psi * s_theta + c_psi * c_theta,
             -c_psi * s_phi * s_theta - c_theta * s_psi,
             c_phi * s_theta],
            [c_phi * s_psi, c_phi * c_psi, s_phi],
            [-c_theta * s_phi * s_psi - c_psi * s_theta,
             -c_psi * c_theta * s_phi + s_psi * s_theta,
             c_phi * c_theta],
        ])

        np.testing.assert_allclose(xtrack_matrix, rot.as_matrix(), atol=1e-14)

    def test_at_and_xsuite_conversions_are_backend_specific(self):
        rot = at_rotation(pitch=0.11, yaw=-0.07, roll=0.05)

        at_roll, at_pitch, at_yaw = at_angles_from_rotation(rot)
        xs_rot_s, xs_rot_x, xs_rot_y = xsuite_angles_from_rotation(rot)

        assert at_roll == pytest.approx(0.05)
        assert at_pitch == pytest.approx(0.11)
        assert at_yaw == pytest.approx(-0.07)
        assert xs_rot_s != pytest.approx(at_roll)
        assert xs_rot_x != pytest.approx(at_pitch)
        assert xs_rot_y != pytest.approx(at_yaw)


def _make_element(length=0.0, bending_angle=0.0):
    return SimpleNamespace(
        Length=length,
        BendingAngle=bending_angle,
        R1=None,
        R2=None,
        T1=None,
        T2=None,
    )


class TestUpdateATTransformation:
    def test_update_at_transformation_zero_offsets(self):
        elem = _make_element(length=1.0)
        result = update_at_transformation(elem, dx=0, dy=0, ds=0)
        np.testing.assert_allclose(result.T1, np.zeros(6), atol=1e-14)
        np.testing.assert_allclose(result.T2, np.zeros(6), atol=1e-14)
        np.testing.assert_allclose(result.R1, np.eye(6), atol=1e-14)
        np.testing.assert_allclose(result.R2, np.eye(6), atol=1e-14)

    def test_update_at_transformation_dx_only(self):
        dx_val = 0.001
        elem = _make_element(length=1.0)
        result = update_at_transformation(elem, dx=dx_val, dy=0, ds=0)

        assert abs(result.T1[0]) > 1e-6
        assert abs(result.T2[0]) > 1e-6
        np.testing.assert_allclose(result.T1[2], 0.0, atol=1e-14)
        np.testing.assert_allclose(result.T2[2], 0.0, atol=1e-14)
        np.testing.assert_allclose(result.R1, np.eye(6), atol=1e-14)
        np.testing.assert_allclose(result.R2, np.eye(6), atol=1e-14)

    def test_update_at_transformation_roll_only(self):
        elem = _make_element(length=1.0)
        result = update_at_transformation(elem, dx=0, dy=0, ds=0, rot=at_rotation_matrix(roll=0.01))

        assert not np.allclose(result.R1, np.eye(6), atol=1e-6)
        assert not np.allclose(result.R2, np.eye(6), atol=1e-6)
        np.testing.assert_allclose(result.T1, np.zeros(6), atol=1e-14)
        np.testing.assert_allclose(result.T2, np.zeros(6), atol=1e-14)

    def test_update_at_transformation_with_dipole(self):
        angle = 0.05
        length = 2.0
        dx_val = 0.001
        elem = _make_element(length=length, bending_angle=angle)
        result = update_at_transformation(elem, dx=dx_val, dy=0, ds=0)

        elem_drift = _make_element(length=length, bending_angle=0.0)
        result_drift = update_at_transformation(elem_drift, dx=dx_val, dy=0, ds=0)

        assert not np.allclose(result.T2, result_drift.T2, atol=1e-8)
        assert not np.allclose(result.R2, result_drift.R2, atol=1e-8)
        np.testing.assert_allclose(result.T1, result_drift.T1, atol=1e-14)
        np.testing.assert_allclose(result.R1, result_drift.R1, atol=1e-14)

    def test_update_at_transformation_modifies_element_in_place(self):
        elem = _make_element(length=1.0)
        result = update_at_transformation(elem, dx=0.001, dy=0, ds=0)
        assert result is elem
