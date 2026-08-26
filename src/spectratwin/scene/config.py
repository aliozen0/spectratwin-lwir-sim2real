"""Scene sampling configuration (SPEC-003 "scene config" input).

Every randomization range is bounded and explicit, per SPEC-003's
requirement that "randomization ranges MUST be configurable and bounded."
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spectratwin.real_data.taxonomy import PROJECT_CATEGORIES

SCENE_CONFIG_SCHEMA_VERSION = "spectratwin-scene-config-v1"

#: SPEC-003 goals: "one urban intersection/road scene family." A single
#: supported layout, not a general road-network generator (see Non-goals:
#: no city-scale GIS).
RoadLayout = Literal["four_way_intersection"]


class ObjectCountPrior(BaseModel):
    """Inclusive [min_count, max_count] sampling range for one category."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_count: int = Field(ge=0)
    max_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _range_is_valid(self) -> ObjectCountPrior:
        if self.max_count < self.min_count:
            raise ValueError(
                f"max_count ({self.max_count}) must be >= min_count ({self.min_count})"
            )
        return self


class SceneConfig(BaseModel):
    """Bounded, deterministic scene-sampling configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = SCENE_CONFIG_SCHEMA_VERSION
    road_layout: RoadLayout = "four_way_intersection"
    road_arm_length_m: float = Field(gt=0)
    road_width_m: float = Field(gt=0)
    sidewalk_width_m: float = Field(gt=0)
    object_count_priors: dict[str, ObjectCountPrior]
    min_clearance_m: float = Field(ge=0)
    orientation_jitter_rad: float = Field(ge=0, le=3.141592653589793)
    placement_retry_budget: int = Field(gt=0)

    @model_validator(mode="after")
    def _priors_use_known_categories(self) -> SceneConfig:
        unknown = sorted(set(self.object_count_priors) - set(PROJECT_CATEGORIES))
        if unknown:
            raise ValueError(
                f"object_count_priors has unknown categories {unknown}; "
                f"must be a subset of {PROJECT_CATEGORIES}"
            )
        return self
