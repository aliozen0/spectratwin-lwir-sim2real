"""Pinhole camera intrinsics: HFOV/focal-length representation (SPEC-004).

Convention: pixel origin at the top-left corner, u right, v down (matches
the image-array convention already used by ``FlirSampleRecord``/bbox data).
Square pixels are assumed (``fx == fy``); the default benchmark-informed
target is 640x512 at roughly 45 deg HFOV, but this module does not claim
calibration to any specific physical sensor (e.g. FLIR Tau 2).
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

INTRINSICS_SCHEMA_VERSION = "spectratwin-camera-intrinsics-v1"


class DistortionParameters(BaseModel):
    """Stored but unapplied here; SPEC-008 sensor pipeline applies distortion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0


class CameraIntrinsics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = INTRINSICS_SCHEMA_VERSION
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    hfov_deg: float = Field(gt=0, lt=180)
    focal_length_px: float = Field(gt=0)
    principal_point_px: tuple[float, float]
    distortion: DistortionParameters = DistortionParameters()

    @model_validator(mode="after")
    def _principal_point_within_image(self) -> CameraIntrinsics:
        cx, cy = self.principal_point_px
        if not (0.0 <= cx <= self.width_px):
            raise ValueError(f"principal_point x={cx} outside [0, {self.width_px}]")
        if not (0.0 <= cy <= self.height_px):
            raise ValueError(f"principal_point y={cy} outside [0, {self.height_px}]")
        return self

    def matrix(self) -> np.ndarray:
        """3x3 pinhole intrinsic matrix ``K`` (square-pixel, zero skew)."""
        cx, cy = self.principal_point_px
        return np.array(
            [
                [self.focal_length_px, 0.0, cx],
                [0.0, self.focal_length_px, cy],
                [0.0, 0.0, 1.0],
            ]
        )

    def vfov_deg(self) -> float:
        """Vertical FOV implied by the shared focal length and image height."""
        return focal_px_to_fov_deg(self.height_px, self.focal_length_px)


def hfov_deg_to_focal_px(width_px: int, hfov_deg: float) -> float:
    """Pinhole relation: ``focal_px = (width_px / 2) / tan(hfov / 2)``."""
    if width_px <= 0:
        raise ValueError("width_px must be positive")
    if not (0.0 < hfov_deg < 180.0):
        raise ValueError("hfov_deg must be in (0, 180)")
    half_fov_rad = math.radians(hfov_deg) / 2.0
    return (width_px / 2.0) / math.tan(half_fov_rad)


def focal_px_to_fov_deg(extent_px: int, focal_px: float) -> float:
    """Inverse of :func:`hfov_deg_to_focal_px` (works for width or height)."""
    if extent_px <= 0:
        raise ValueError("extent_px must be positive")
    if focal_px <= 0:
        raise ValueError("focal_px must be positive")
    return math.degrees(2.0 * math.atan((extent_px / 2.0) / focal_px))


def build_intrinsics(
    *,
    width_px: int,
    height_px: int,
    hfov_deg: float,
    principal_point_px: tuple[float, float] | None = None,
    distortion: DistortionParameters | None = None,
) -> CameraIntrinsics:
    """Derive a full :class:`CameraIntrinsics` from width/height/HFOV.

    ``principal_point_px`` defaults to the exact image center
    ``(width_px / 2, height_px / 2)``.
    """
    focal_px = hfov_deg_to_focal_px(width_px, hfov_deg)
    return CameraIntrinsics(
        width_px=width_px,
        height_px=height_px,
        hfov_deg=hfov_deg,
        focal_length_px=focal_px,
        principal_point_px=principal_point_px or (width_px / 2.0, height_px / 2.0),
        distortion=distortion or DistortionParameters(),
    )
