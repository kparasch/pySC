"""Coordinate and rotation helpers for pySC core misalignments."""

import numpy as np
from scipy.spatial.transform import Rotation

EPS = 1e-12


def at_rotation(pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> Rotation:
    """
    Return the AT/pySC rotation convention as a SciPy Rotation.

    pySC stores element angles in AT convention: pitch around x, yaw around y,
    and roll/tilt around s/z. In SciPy this is ``from_euler('zyx',
    [roll, yaw, pitch])``, whose matrix is ``Rx(pitch) @ Ry(yaw) @ Rz(roll)``
    in standard right-handed coordinates.
    """
    return Rotation.from_euler('zyx', [roll, yaw, pitch])


def at_rotation_matrix(pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> np.ndarray:
    return at_rotation(pitch=pitch, yaw=yaw, roll=roll).as_matrix()


def as_rotation(rot) -> Rotation:
    if isinstance(rot, Rotation):
        return rot
    return Rotation.from_matrix(rot)


def at_angles_from_rotation(rot) -> tuple[float, float, float]:
    """Convert an AT/pySC-convention rotation to ``(roll, pitch, yaw)``."""
    roll, yaw, pitch = as_rotation(rot).as_euler('zyx')
    return float(roll), float(pitch), float(yaw)


def xsuite_angles_from_rotation(rot) -> tuple[float, float, float]:
    """
    Convert an AT/pySC rotation matrix to XSuite element misalignment angles.

    XTrack applies element rotations as ``Ry(rot_y_rad) @ Rx(-rot_x_rad) @
    Rz(rot_s_rad_no_frame)`` in standard right-handed matrix form. The negative
    x sign follows XSuite's convention that positive ``rot_x_rad`` rotates
    positive s toward positive y.

    Returns
    -------
    tuple
        ``(rot_s_rad_no_frame, rot_x_rad, rot_y_rad)``.
    """
    matrix = as_rotation(rot).as_matrix()
    rot_x_rad = np.arcsin(np.clip(matrix[1, 2], -1.0, 1.0))
    rot_y_rad = np.arctan2(matrix[0, 2], matrix[2, 2])
    rot_s_rad_no_frame = np.arctan2(matrix[1, 0], matrix[1, 1])
    return float(rot_s_rad_no_frame), float(rot_x_rad), float(rot_y_rad)


def axis_angle_rotation(axis, angle) -> Rotation:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < EPS or abs(angle) < EPS:
        return Rotation.identity()
    return Rotation.from_rotvec(axis / norm * angle)


def rotation_from_vectors(source, target) -> Rotation:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source_norm = np.linalg.norm(source)
    target_norm = np.linalg.norm(target)
    if source_norm < EPS or target_norm < EPS:
        return Rotation.identity()

    a = source / source_norm
    b = target / target_norm
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    cross_norm = np.linalg.norm(cross)

    if cross_norm < EPS:
        if dot > 0:
            return Rotation.identity()
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < EPS:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        return axis_angle_rotation(axis, np.pi)

    return axis_angle_rotation(cross, np.arctan2(cross_norm, dot))
