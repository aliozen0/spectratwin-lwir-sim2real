"""Deterministic static geometry for the four-way-intersection scene family.

All coordinates are meters in a top-down (x, y) plane centered on the
intersection. This module has no Blender dependency: it is the pure-data
layer SPEC-003 separates from "Blender scene realization at adapter layer"
(docs/ARCHITECTURE.md "Scene generation must not directly write datasets").
"""

from __future__ import annotations

from typing import NamedTuple

from spectratwin.scene.config import SceneConfig


class Rect(NamedTuple):
    """Axis-aligned rectangle: inclusive [x_min, x_max] x [y_min, y_max]."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def contains_rect(self, other: Rect) -> bool:
        return (
            self.x_min <= other.x_min
            and other.x_max <= self.x_max
            and self.y_min <= other.y_min
            and other.y_max <= self.y_max
        )


def rects_overlap(a: Rect, b: Rect) -> bool:
    """True if two axis-aligned rects share any positive-area overlap."""
    return a.x_min < b.x_max and b.x_min < a.x_max and a.y_min < b.y_max and b.y_min < a.y_max


class RoadNetwork(NamedTuple):
    """The road/sidewalk regions and lane-orientation hints for one scene."""

    road_rects: tuple[Rect, ...]
    sidewalk_rects: tuple[Rect, ...]
    #: One nominal travel-direction angle (radians) per road rect, same
    #: index alignment as ``road_rects`` — used to orient cars/bicycles
    #: along their arm instead of at an arbitrary angle.
    road_orientations_rad: tuple[float, ...]


def build_four_way_intersection(config: SceneConfig) -> RoadNetwork:
    """Two perpendicular road arms crossing at the origin, each flanked by sidewalks."""
    half_length = config.road_arm_length_m / 2.0
    half_road = config.road_width_m / 2.0
    sidewalk_width = config.sidewalk_width_m

    horizontal_road = Rect(-half_length, half_length, -half_road, half_road)
    vertical_road = Rect(-half_road, half_road, -half_length, half_length)

    horizontal_sidewalk_north = Rect(
        -half_length, half_length, half_road, half_road + sidewalk_width
    )
    horizontal_sidewalk_south = Rect(
        -half_length, half_length, -half_road - sidewalk_width, -half_road
    )
    vertical_sidewalk_east = Rect(half_road, half_road + sidewalk_width, -half_length, half_length)
    vertical_sidewalk_west = Rect(
        -half_road - sidewalk_width, -half_road, -half_length, half_length
    )

    return RoadNetwork(
        road_rects=(horizontal_road, vertical_road),
        sidewalk_rects=(
            horizontal_sidewalk_north,
            horizontal_sidewalk_south,
            vertical_sidewalk_east,
            vertical_sidewalk_west,
        ),
        # Horizontal arm runs along x (0 rad), vertical arm along y (pi/2 rad).
        road_orientations_rad=(0.0, 1.5707963267948966),
    )
