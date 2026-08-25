import math
from typing import Any

import pytest
from pydantic import ValidationError

from spectratwin.scene.assets import AssetDescriptor, build_asset_registry
from spectratwin.scene.config import ObjectCountPrior, SceneConfig
from spectratwin.scene.placement import PlacedObject, PlacementIssue
from spectratwin.scene.report import summarize_scene_distribution
from spectratwin.scene.scene import SceneDescription, sample_scene, validate_scene


def _registry():
    return build_asset_registry(
        [
            AssetDescriptor(
                asset_id="person-01",
                category="person",
                footprint_length_m=0.6,
                footprint_width_m=0.6,
                license_id="cc0",
            ),
            AssetDescriptor(
                asset_id="car-01",
                category="car",
                footprint_length_m=4.5,
                footprint_width_m=1.8,
                license_id="cc0",
            ),
            AssetDescriptor(
                asset_id="bicycle-01",
                category="bicycle",
                footprint_length_m=1.8,
                footprint_width_m=0.6,
                license_id="cc0",
            ),
        ]
    )


def _config(**overrides: Any) -> SceneConfig:
    defaults: dict[str, Any] = dict(
        road_arm_length_m=60.0,
        road_width_m=6.0,
        sidewalk_width_m=2.0,
        object_count_priors={
            "car": ObjectCountPrior(min_count=1, max_count=5),
            "person": ObjectCountPrior(min_count=0, max_count=5),
            "bicycle": ObjectCountPrior(min_count=0, max_count=3),
        },
        min_clearance_m=0.3,
        orientation_jitter_rad=0.2,
        placement_retry_budget=30,
    )
    defaults.update(overrides)
    return SceneConfig(**defaults)


# --- Asset registry ---------------------------------------------------------


def test_build_asset_registry_rejects_duplicate_asset_ids():
    with pytest.raises(ValidationError, match="unique"):
        build_asset_registry(
            [
                AssetDescriptor(
                    asset_id="dup",
                    category="car",
                    footprint_length_m=1.0,
                    footprint_width_m=1.0,
                    license_id="cc0",
                ),
                AssetDescriptor(
                    asset_id="dup",
                    category="person",
                    footprint_length_m=1.0,
                    footprint_width_m=1.0,
                    license_id="cc0",
                ),
            ]
        )


def test_asset_descriptor_rejects_unknown_category():
    with pytest.raises(ValidationError, match="category"):
        AssetDescriptor(
            asset_id="x",
            category="truck",
            footprint_length_m=1.0,
            footprint_width_m=1.0,
            license_id="cc0",
        )


def test_asset_registry_fingerprint_is_stable_and_content_sensitive():
    registry_a = _registry()
    registry_b = _registry()
    assert registry_a.fingerprint == registry_b.fingerprint

    changed = build_asset_registry(
        [*registry_a.assets[:-1], registry_a.assets[-1].model_copy(update={"license_id": "other"})]
    )
    assert changed.fingerprint != registry_a.fingerprint


# --- Scene config ------------------------------------------------------------


def test_scene_config_rejects_inverted_count_range():
    with pytest.raises(ValidationError, match="max_count"):
        _config(object_count_priors={"car": ObjectCountPrior(min_count=5, max_count=1)})


def test_scene_config_rejects_unknown_prior_category():
    with pytest.raises(ValidationError, match="unknown categories"):
        SceneConfig(
            road_arm_length_m=60.0,
            road_width_m=6.0,
            sidewalk_width_m=2.0,
            object_count_priors={"truck": ObjectCountPrior(min_count=0, max_count=1)},
            min_clearance_m=0.3,
            orientation_jitter_rad=0.2,
            placement_retry_budget=30,
        )


def test_scene_config_rejects_non_positive_geometry():
    with pytest.raises(ValidationError):
        _config(road_arm_length_m=0.0)


# --- Deterministic sampling ---------------------------------------------------


def test_sample_scene_is_deterministic_for_same_seed():
    config = _config()
    registry = _registry()

    scene_a = sample_scene(config=config, asset_registry=registry, sample_seed=7)
    scene_b = sample_scene(config=config, asset_registry=registry, sample_seed=7)

    assert scene_a == scene_b


def test_sample_scene_differs_across_seeds():
    config = _config()
    registry = _registry()

    scenes = [
        sample_scene(config=config, asset_registry=registry, sample_seed=seed) for seed in range(5)
    ]
    serialized = {scene.model_dump_json() for scene in scenes}
    assert len(serialized) == 5


def test_sample_scene_rejects_negative_seed():
    with pytest.raises(ValueError, match="non-negative"):
        sample_scene(config=_config(), asset_registry=_registry(), sample_seed=-1)


def test_100_sampled_scenes_validate_with_no_illegal_placement():
    config = _config()
    registry = _registry()

    for seed in range(100):
        scene = sample_scene(config=config, asset_registry=registry, sample_seed=seed)
        violations = validate_scene(scene, registry)
        assert violations == (), f"seed={seed}: {violations}"


def _oriented_corners(obj) -> list[tuple[float, float]]:
    """Four world-space corners of an object's true rotated footprint."""
    cx, cy = obj.position_m
    cos_t = math.cos(obj.orientation_rad)
    sin_t = math.sin(obj.orientation_rad)
    half_length = obj.footprint_length_m / 2.0
    half_width = obj.footprint_width_m / 2.0
    corners = []
    for sign_length, sign_width in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        local_x = sign_length * half_length
        local_y = sign_width * half_width
        corners.append(
            (cx + local_x * cos_t - local_y * sin_t, cy + local_x * sin_t + local_y * cos_t)
        )
    return corners


def _oriented_rectangles_overlap(a, b) -> bool:
    """Independent separating-axis test on the true rotated footprints.

    Deliberately does NOT reuse ``PlacedObject.footprint_rect``: this is the
    physical requirement ("no impossible placements"), so it must be able to
    disagree with the implementation rather than mirror it.
    """
    for polygon in (_oriented_corners(a), _oriented_corners(b)):
        for i in range(len(polygon)):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % len(polygon)]
            # Outward normal of this edge.
            axis = (-(y2 - y1), x2 - x1)
            norm = math.hypot(*axis)
            axis = (axis[0] / norm, axis[1] / norm)

            projections_a = [axis[0] * px + axis[1] * py for px, py in _oriented_corners(a)]
            projections_b = [axis[0] * px + axis[1] * py for px, py in _oriented_corners(b)]
            if max(projections_a) <= min(projections_b) or max(projections_b) <= min(projections_a):
                return False  # separating axis found
    return True


def test_sampled_scenes_have_no_physically_overlapping_objects():
    """SPEC-003 goal 'no impossible placements', checked against real geometry.

    Uses an independent oriented-rectangle SAT oracle, so a footprint model
    that ignores orientation cannot make this test pass by agreeing with
    itself.
    """
    config = _config()
    registry = _registry()

    for seed in range(100):
        scene = sample_scene(config=config, asset_registry=registry, sample_seed=seed)
        for i in range(len(scene.objects)):
            for j in range(i + 1, len(scene.objects)):
                a, b = scene.objects[i], scene.objects[j]
                assert not _oriented_rectangles_overlap(a, b), (
                    f"seed={seed}: {a.category} at {a.position_m} "
                    f"physically overlaps {b.category} at {b.position_m}"
                )


def test_footprint_rect_tracks_orientation():
    """A rotated object's world-axis extents must follow its heading."""
    east_facing = PlacedObject(
        asset_id="car-01",
        category="car",
        position_m=(0.0, 0.0),
        orientation_rad=0.0,
        footprint_length_m=4.0,
        footprint_width_m=2.0,
    )
    north_facing = east_facing.model_copy(update={"orientation_rad": math.pi / 2})

    east_rect = east_facing.footprint_rect(0.0)
    north_rect = north_facing.footprint_rect(0.0)

    # Facing east: 4 m along x, 2 m along y. Facing north: swapped.
    assert east_rect.x_max - east_rect.x_min == pytest.approx(4.0)
    assert east_rect.y_max - east_rect.y_min == pytest.approx(2.0)
    assert north_rect.x_max - north_rect.x_min == pytest.approx(2.0)
    assert north_rect.y_max - north_rect.y_min == pytest.approx(4.0)


# --- Bounded placement failure -------------------------------------------------


def test_placement_retry_budget_exhaustion_is_bounded_not_infinite():
    # A tiny world with far more objects requested than can physically fit.
    tight_config = _config(
        road_arm_length_m=4.0,
        road_width_m=3.0,
        sidewalk_width_m=1.0,
        object_count_priors={"car": ObjectCountPrior(min_count=50, max_count=50)},
        placement_retry_budget=5,
    )
    registry = _registry()

    scene = sample_scene(config=tight_config, asset_registry=registry, sample_seed=0)

    assert len(scene.objects) < 50
    assert any(issue.category == "car" for issue in scene.placement_issues)
    for issue in scene.placement_issues:
        assert issue.placed_count <= issue.requested_count


def test_placement_reports_typed_issue_for_category_missing_from_registry():
    config = _config(object_count_priors={"bicycle": ObjectCountPrior(min_count=1, max_count=1)})
    empty_registry = build_asset_registry([])

    scene = sample_scene(config=config, asset_registry=empty_registry, sample_seed=0)

    assert scene.objects == ()
    assert scene.placement_issues == (
        PlacementIssue(
            category="bicycle",
            requested_count=1,
            placed_count=0,
            reason="no_registered_asset_for_category",
        ),
    )


# --- Distribution report -------------------------------------------------------


def test_summarize_scene_distribution_computes_counts_and_distances():
    scene = SceneDescription(
        sample_seed=0,
        config_fingerprint="0" * 64,
        asset_registry_fingerprint="0" * 64,
        road_layout="four_way_intersection",
        road_arm_length_m=60.0,
        road_width_m=6.0,
        sidewalk_width_m=2.0,
        min_clearance_m=0.3,
        objects=(
            PlacedObject(
                asset_id="car-01",
                category="car",
                position_m=(3.0, 0.0),
                orientation_rad=0.0,
                footprint_length_m=4.5,
                footprint_width_m=1.8,
            ),
            PlacedObject(
                asset_id="car-01",
                category="car",
                position_m=(0.0, 4.0),
                orientation_rad=0.0,
                footprint_length_m=4.5,
                footprint_width_m=1.8,
            ),
        ),
        placement_issues=(),
    )

    report = summarize_scene_distribution([scene])

    assert report.scene_count == 1
    assert report.object_count_by_category == {"car": 2}
    assert report.placement_issue_count == 0
    stats = report.distance_by_category["car"]
    assert stats.count == 2
    assert stats.min_distance_m == pytest.approx(3.0)
    assert stats.max_distance_m == pytest.approx(4.0)
    assert stats.mean_distance_m == pytest.approx(3.5)
    assert math.hypot(3.0, 0.0) == pytest.approx(3.0)


def test_summarize_scene_distribution_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one scene"):
        summarize_scene_distribution([])
