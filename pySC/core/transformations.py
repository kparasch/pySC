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


def _translation_vector(ld, r3d, xaxis_xyz, yaxis_xyz, offsets):
    tD0 = np.array([-np.dot(offsets, xaxis_xyz), 0, -np.dot(offsets, yaxis_xyz), 0, 0, 0])
    T0 = np.array([ld * r3d[2, 0] / r3d[2, 2], r3d[2, 0],
                   ld * r3d[2, 1] / r3d[2, 2], r3d[2, 1],
                   0, ld / r3d[2, 2]])
    return T0 + tD0


def _r_matrix(ld, r3d):
    return np.array([
        [r3d[1, 1] / r3d[2, 2], ld * r3d[1, 1] / r3d[2, 2] ** 2,
         -r3d[0, 1] / r3d[2, 2], -ld * r3d[0, 1] / r3d[2, 2] ** 2, 0, 0],
        [0, r3d[0, 0], 0, r3d[1, 0], r3d[2, 0], 0],
        [-r3d[1, 0] / r3d[2, 2], -ld * r3d[1, 0] / r3d[2, 2] ** 2,
         r3d[0, 0] / r3d[2, 2], ld * r3d[0, 0] / r3d[2, 2] ** 2, 0, 0],
        [0, r3d[0, 1], 0, r3d[1, 1], r3d[2, 1], 0],
        [0, 0, 0, 0, 1, 0],
        [-r3d[0, 2] / r3d[2, 2], -ld * r3d[0, 2] / r3d[2, 2] ** 2,
         -r3d[1, 2] / r3d[2, 2], -ld * r3d[1, 2] / r3d[2, 2] ** 2, 0, 1],
    ])


def update_at_transformation(element, dx=0.0, dy=0.0, ds=0.0, rot=None):
    """
    Update AT element T/R matrices from pySC local offsets and a SciPy rotation.
    """
    mag_length = getattr(element, "Length", 0)
    mag_theta = getattr(element, 'BendingAngle', 0)
    offsets = np.array([dx, dy, ds])

    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    r_3d = as_rotation(rot if rot is not None else Rotation.identity()).as_matrix()
    ld = np.dot(np.dot(r_3d, z_axis), offsets)

    T = _translation_vector(ld, r_3d, np.dot(r_3d, x_axis), np.dot(r_3d, y_axis), offsets)
    element.R1 = _r_matrix(ld, r_3d)
    element.T1 = np.dot(np.linalg.inv(element.R1), T)

    RX = r_3d
    RB = at_rotation_matrix(yaw=-mag_theta)
    r_3d = np.dot(RB.T, np.dot(RX.T, RB))
    OPp = np.array([(mag_length * (np.cos(mag_theta) - 1) / mag_theta if mag_theta else 0),
                    0,
                    mag_length * (np.sin(mag_theta) / mag_theta if mag_theta else 1)])

    OpPp = OPp - np.dot(RX, OPp) - offsets
    ld = np.dot(np.dot(RB, z_axis), OpPp)

    element.T2 = _translation_vector(ld, r_3d, np.dot(RB, x_axis), np.dot(RB, y_axis), OpPp)
    element.R2 = _r_matrix(ld, r_3d)
    return element
