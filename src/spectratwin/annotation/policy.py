"""Persisted annotation inclusion policy (SPEC-005)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ANNOTATION_POLICY_SCHEMA_VERSION = "spectratwin-annotation-policy-v1"


class ExclusionReason(StrEnum):
    """Stable reasons why an expected object has no COCO annotation."""

    ZERO_VISIBLE_AREA = "zero_visible_area"
    BELOW_VISIBLE_AREA = "below_visible_area"
    BELOW_BBOX_SIDE = "below_bbox_side"


class AnnotationPolicy(BaseModel):
    """Resolved thresholds persisted with every annotated image."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["spectratwin-annotation-policy-v1"] = ANNOTATION_POLICY_SCHEMA_VERSION
    min_visible_area_px: int = Field(default=1, ge=1, strict=True)
    min_bbox_side_px: int = Field(default=1, ge=1, strict=True)

    def exclusion_reason(
        self,
        *,
        visible_area_px: int,
        bbox_xywh: tuple[int, int, int, int] | None,
    ) -> ExclusionReason | None:
        """Return the policy reason for exclusion, or ``None`` when included."""
        if visible_area_px < 0:
            raise ValueError("visible_area_px must be non-negative")
        if visible_area_px == 0:
            if bbox_xywh is not None:
                raise ValueError("zero visible area must not have a bbox")
            return ExclusionReason.ZERO_VISIBLE_AREA
        if bbox_xywh is None:
            raise ValueError("positive visible area requires a bbox")
        if visible_area_px < self.min_visible_area_px:
            return ExclusionReason.BELOW_VISIBLE_AREA
        _, _, width_px, height_px = bbox_xywh
        if width_px < self.min_bbox_side_px or height_px < self.min_bbox_side_px:
            return ExclusionReason.BELOW_BBOX_SIDE
        return None
