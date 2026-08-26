"""Explicit world/camera transform convention and pinhole projection (SPEC-004).

World frame: x right (east), y forward (north), z up, meters.

Camera frame (OpenCV-style, matches ``intrinsics.py``'s pixel convention):
x right, y down, z forward (the optical axis). At ``pitch_deg=0,
yaw_deg=0`` the camera looks along world +y with its optical axis level.

Sign conventions, verified by tests in ``tests/unit/test_camera.py``:

- positive ``pitch_deg`` tilts the optical axis **downward** (toward -z);
- positive ``yaw_deg`` is a right-handed rotation about world +z, which
  turns the optical axis from north (+y) toward **west** (-x); at
  ``yaw_deg=90`` the camera looks along world -x.

Roll is fixed at zero (see ``pose.py``).
"""

from __future__ import annotations

import math

import numpy as np

from spectratwin.camera.intrinsics import CameraIntrinsics
from spectratwin.camera.pose import CameraPose

#: Camera axes expressed in world coordinates at pitch=0, yaw=0: columns are
#: (camera-right, camera-down, camera-forward) in world (x, y, z).
_BASE_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]
)


def _rotation_about_world_x(theta_rad: float) -> np.ndarray:
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_t, -sin_t],
            [0.0, sin_t, cos_t],
        ]
    )


def _rotation_about_world_z(theta_rad: float) -> np.ndarray:
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    return np.array(
        [
            [cos_t, -sin_t, 0.0],
            [sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def rotation_matrix(pose: CameraPose) -> np.ndarray:
    """3x3 rotation mapping a camera-frame vector to a world-frame vector."""
    pitch_rad = math.radians(pose.pitch_deg)
    yaw_rad = math.radians(pose.yaw_deg)
    # Pitch about world x while still north-facing, then yaw about world z.
    # Negated pitch: positive pitch_deg must tilt the optical axis down.
    pitch_rotation = _rotation_about_world_x(-pitch_rad)
    yaw_rotation = _rotation_about_world_z(yaw_rad)
    return yaw_rotation @ pitch_rotation @ _BASE_ROTATION


def world_to_camera(pose: CameraPose, point_world: tuple[float, float, float]) -> np.ndarray:
    """Transform a world-frame point into the camera frame."""
    rotation = rotation_matrix(pose)
    translated = np.array(point_world) - np.array(pose.position_m)
    return rotation.T @ translated


def camera_to_world(pose: CameraPose, point_camera: np.ndarray) -> np.ndarray:
    """Inverse of :func:`world_to_camera`."""
    rotation = rotation_matrix(pose)
    return rotation @ np.asarray(point_camera, dtype=float) + np.array(pose.position_m)


def project_to_pixel(
    intrinsics: CameraIntrinsics, point_camera: np.ndarray
) -> tuple[float, float] | None:
    """Pinhole projection of a camera-frame point; ``None`` if behind the camera."""
    x, y, z = point_camera
    if z <= 0:
        return None
    cx, cy = intrinsics.principal_point_px
    u = intrinsics.focal_length_px * (x / z) + cx
    v = intrinsics.focal_length_px * (y / z) + cy
    return float(u), float(v)


def project_world_point(
    intrinsics: CameraIntrinsics, pose: CameraPose, point_world: tuple[float, float, float]
) -> tuple[float, float] | None:
    """Convenience: world point -> camera frame -> pixel, in one call."""
    return project_to_pixel(intrinsics, world_to_camera(pose, point_world))
