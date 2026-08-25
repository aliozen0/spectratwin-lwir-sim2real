"""Bounded, deterministic camera pose sampling (SPEC-004).

World convention (matches ``spectratwin.scene`` ground plane): x right, y
forward, z up, all in meters. Camera orientation is roll-free (SPEC-004's
Requirements list only "height/pitch/yaw/location" — roll is a documented,
reversible omission: a level-horizon fixed-mount camera).
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

POSE_SCHEMA_VERSION = "spectratwin-camera-pose-v1"


class BoundedRange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_value: float
    max_value: float

    @model_validator(mode="after")
    def _range_is_valid(self) -> BoundedRange:
        if self.max_value < self.min_value:
            raise ValueError(
                f"max_value ({self.max_value}) must be >= min_value ({self.min_value})"
            )
        return self


class CameraPoseConfig(BaseModel):
    """Bounded sampling ranges for every pose degree of freedom."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    location_x_range_m: BoundedRange
    location_y_range_m: BoundedRange
    height_range_m: BoundedRange
    pitch_deg_range: BoundedRange = Field(description="0 = level, positive tilts down")
    yaw_deg_range: BoundedRange

    @model_validator(mode="after")
    def _height_is_positive(self) -> CameraPoseConfig:
        if self.height_range_m.min_value <= 0:
            raise ValueError("height_range_m.min_value must be > 0 (camera above ground)")
        return self

    @model_validator(mode="after")
    def _pitch_and_yaw_are_bounded_angles(self) -> CameraPoseConfig:
        for name, bound in (
            ("pitch_deg_range", self.pitch_deg_range),
            ("yaw_deg_range", self.yaw_deg_range),
        ):
            if not (-180.0 <= bound.min_value <= 180.0 and -180.0 <= bound.max_value <= 180.0):
                raise ValueError(f"{name} must stay within [-180, 180] degrees")
        return self


class CameraPose(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = POSE_SCHEMA_VERSION
    sample_seed: int = Field(ge=0)
    position_m: tuple[float, float, float]
    pitch_deg: float
    yaw_deg: float


def sample_camera_pose(config: CameraPoseConfig, sample_seed: int) -> CameraPose:
    """Deterministically sample one pose from ``config``/``sample_seed`` alone."""
    if sample_seed < 0:
        raise ValueError("sample_seed must be non-negative")

    rng = np.random.default_rng(sample_seed)
    x = float(rng.uniform(config.location_x_range_m.min_value, config.location_x_range_m.max_value))
    y = float(rng.uniform(config.location_y_range_m.min_value, config.location_y_range_m.max_value))
    z = float(rng.uniform(config.height_range_m.min_value, config.height_range_m.max_value))
    pitch = float(rng.uniform(config.pitch_deg_range.min_value, config.pitch_deg_range.max_value))
    yaw = float(rng.uniform(config.yaw_deg_range.min_value, config.yaw_deg_range.max_value))

    return CameraPose(
        sample_seed=sample_seed,
        position_m=(x, y, z),
        pitch_deg=pitch,
        yaw_deg=yaw,
    )
