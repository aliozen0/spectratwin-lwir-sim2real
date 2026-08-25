"""Deterministic, retry-bounded object placement (SPEC-003).

Placement never loops unboundedly: each object gets at most
``config.placement_retry_budget`` sampling attempts, and a category that
cannot reach its sampled target count records a typed, bounded
:class:`PlacementIssue` instead of raising or spinning.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from spectratwin.scene.assets import AssetRegistry
from spectratwin.scene.config import SceneConfig
from spectratwin.scene.geometry import Rect, RoadNetwork, rects_overlap

#: Which region family each project category is placed into. A documented
#: simplification, not a physical law: pedestrians use sidewalks, wheeled
#: categories use the road.
CATEGORY_REGION: dict[str, str] = {"person": "sidewalk", "car": "road", "bicycle": "road"}


class PlacedObject(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str
    category: str
    position_m: tuple[float, float]
    orientation_rad: float
    footprint_length_m: float = Field(gt=0)
    footprint_width_m: float = Field(gt=0)

    def footprint_rect(self, clearance_m: float) -> Rect:
        """Orientation-aware axis-aligned bound of the rotated footprint.

        The object's length runs along its own ``orientation_rad`` heading,
        so the world-axis extents depend on that heading: a 4.5 m car
        pointing north occupies 4.5 m along y, not along x. This returns
        the tight AABB enclosing that rotated rectangle, expanded by
        ``clearance_m``.

        Collision policy (explicit, per SPEC-003): AABB-of-rotated-footprint
        overlap, not oriented SAT. That is genuinely conservative — the AABB
        contains the true footprint, so any pair this test accepts is truly
        disjoint. It can reject some diagonal near-misses that oriented SAT
        would allow; retry sampling absorbs that.
        """
        half_length = self.footprint_length_m / 2.0
        half_width = self.footprint_width_m / 2.0
        cos_t = abs(math.cos(self.orientation_rad))
        sin_t = abs(math.sin(self.orientation_rad))
        half_extent_x = half_length * cos_t + half_width * sin_t + clearance_m
        half_extent_y = half_length * sin_t + half_width * cos_t + clearance_m
        x, y = self.position_m
        return Rect(x - half_extent_x, x + half_extent_x, y - half_extent_y, y + half_extent_y)


class PlacementIssue(BaseModel):
    """Bounded, typed record of a category that could not be fully placed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: str
    requested_count: int
    placed_count: int
    reason: str = "placement_retry_budget_exhausted"


def _sample_point_in_rects(
    rects: tuple[Rect, ...], rng: np.random.Generator
) -> tuple[float, float, int]:
    """Pick one rect area-weighted, then a uniform point inside it."""
    areas = np.array([rect.area for rect in rects], dtype=float)
    weights = areas / areas.sum()
    rect_index = int(rng.choice(len(rects), p=weights))
    rect = rects[rect_index]
    x = float(rng.uniform(rect.x_min, rect.x_max))
    y = float(rng.uniform(rect.y_min, rect.y_max))
    return x, y, rect_index


def place_objects(
    *,
    config: SceneConfig,
    asset_registry: AssetRegistry,
    road_network: RoadNetwork,
    rng: np.random.Generator,
) -> tuple[tuple[PlacedObject, ...], tuple[PlacementIssue, ...]]:
    """Sample object counts/positions/orientations for every configured category.

    Iterates categories in sorted order so the same config/registry/seed
    always consumes the RNG stream in the same sequence.
    """
    placed: list[PlacedObject] = []
    issues: list[PlacementIssue] = []

    for category in sorted(config.object_count_priors):
        prior = config.object_count_priors[category]
        available_assets = asset_registry.by_category(category)
        target_count = int(rng.integers(prior.min_count, prior.max_count + 1))

        if not available_assets:
            if target_count > 0:
                issues.append(
                    PlacementIssue(
                        category=category,
                        requested_count=target_count,
                        placed_count=0,
                        reason="no_registered_asset_for_category",
                    )
                )
            continue

        region = CATEGORY_REGION[category]
        rects = road_network.road_rects if region == "road" else road_network.sidewalk_rects

        placed_for_category = 0
        for _ in range(target_count):
            asset = available_assets[int(rng.integers(0, len(available_assets)))]
            placed_this_object = False
            for _attempt in range(config.placement_retry_budget):
                x, y, rect_index = _sample_point_in_rects(rects, rng)
                if region == "road":
                    base_orientation = road_network.road_orientations_rad[
                        rect_index % len(road_network.road_orientations_rad)
                    ]
                    jitter = float(
                        rng.uniform(-config.orientation_jitter_rad, config.orientation_jitter_rad)
                    )
                    orientation = base_orientation + jitter
                else:
                    orientation = float(rng.uniform(-math.pi, math.pi))

                candidate = PlacedObject(
                    asset_id=asset.asset_id,
                    category=category,
                    position_m=(x, y),
                    orientation_rad=orientation,
                    footprint_length_m=asset.footprint_length_m,
                    footprint_width_m=asset.footprint_width_m,
                )
                own_rect = candidate.footprint_rect(0.0)
                if not rects[rect_index].contains_rect(own_rect):
                    # Footprint would spill outside its road/sidewalk
                    # region (e.g. sampled too close to an edge) - reject,
                    # not just an out-of-region center point.
                    continue

                candidate_rect = candidate.footprint_rect(config.min_clearance_m)
                if any(
                    rects_overlap(candidate_rect, other.footprint_rect(config.min_clearance_m))
                    for other in placed
                ):
                    continue

                placed.append(candidate)
                placed_for_category += 1
                placed_this_object = True
                break

            if not placed_this_object:
                # Retry budget for this object is exhausted; stop filling
                # this category rather than burning more bounded attempts
                # against an already-crowded scene.
                break

        if placed_for_category < target_count:
            issues.append(
                PlacementIssue(
                    category=category,
                    requested_count=target_count,
                    placed_count=placed_for_category,
                )
            )

    return tuple(placed), tuple(issues)
