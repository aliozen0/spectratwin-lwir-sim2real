"""Object count/distance distribution report across sampled scenes (SPEC-003).

"Distance" is measured from the intersection origin (0, 0): SPEC-003 has no
camera yet (SPEC-004), so a camera-relative distance cannot be computed
here without inventing an unspecced camera pose.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from spectratwin.scene.scene import SceneDescription


class CategoryDistanceStats(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int
    min_distance_m: float
    max_distance_m: float
    mean_distance_m: float


class SceneDistributionReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scene_count: int
    object_count_by_category: dict[str, int]
    distance_by_category: dict[str, CategoryDistanceStats]
    placement_issue_count: int


def summarize_scene_distribution(scenes: list[SceneDescription]) -> SceneDistributionReport:
    if not scenes:
        raise ValueError("summarize_scene_distribution requires at least one scene")

    distances_by_category: dict[str, list[float]] = {}
    placement_issue_count = 0

    for scene in scenes:
        placement_issue_count += len(scene.placement_issues)
        for obj in scene.objects:
            x, y = obj.position_m
            distance = math.hypot(x, y)
            distances_by_category.setdefault(obj.category, []).append(distance)

    object_count_by_category = {
        category: len(distances) for category, distances in distances_by_category.items()
    }
    distance_by_category = {
        category: CategoryDistanceStats(
            count=len(distances),
            min_distance_m=min(distances),
            max_distance_m=max(distances),
            mean_distance_m=sum(distances) / len(distances),
        )
        for category, distances in distances_by_category.items()
    }

    return SceneDistributionReport(
        scene_count=len(scenes),
        object_count_by_category=object_count_by_category,
        distance_by_category=distance_by_category,
        placement_issue_count=placement_issue_count,
    )
