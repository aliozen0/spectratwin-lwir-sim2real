"""SPEC-007 final acceptance: a SPEC-003 scene renders through the same material
builder as the controlled scene, with no second physics path."""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.renderer


def _tiny_scene():
    from spectratwin.scene.assets import AssetDescriptor, build_asset_registry
    from spectratwin.scene.config import ObjectCountPrior, SceneConfig
    from spectratwin.scene.scene import sample_scene

    registry = build_asset_registry(
        [
            AssetDescriptor(
                asset_id="car-000",
                category="car",
                footprint_length_m=4.5,
                footprint_width_m=1.8,
                license_id="placeholder-primitive",
            ),
            AssetDescriptor(
                asset_id="person-000",
                category="person",
                footprint_length_m=0.6,
                footprint_width_m=0.6,
                license_id="placeholder-primitive",
            ),
        ]
    )
    config = SceneConfig(
        road_arm_length_m=30.0,
        road_width_m=8.0,
        sidewalk_width_m=2.5,
        object_count_priors={
            "car": ObjectCountPrior(min_count=2, max_count=3),
            "person": ObjectCountPrior(min_count=1, max_count=2),
        },
        min_clearance_m=0.5,
        orientation_jitter_rad=0.2,
        placement_retry_budget=64,
    )
    return sample_scene(config=config, asset_registry=registry, sample_seed=5)


def test_scene_renders_a_finite_non_empty_thermal_frame(tmp_path) -> None:
    import numpy as np

    from spectratwin.camera.intrinsics import build_intrinsics
    from spectratwin.render.runtime import configure_scene, render_to_array, reset_scene
    from spectratwin.render.scene_builder import build_camera, build_scene
    from spectratwin.render.settings import RenderSettings

    scene = _tiny_scene()
    settings = RenderSettings.for_reference_check(
        width_px=64, height_px=52, seed=0, diffuse_bounces=1
    )

    reset_scene()
    configure_scene(settings)
    build_scene(scene=scene, ambient_temperature_k=293.15, master_seed=0)
    build_camera(
        build_intrinsics(width_px=64, height_px=52, hfov_deg=45.0),
        location_m=(0.0, -25.0, 3.0),
        rotation_rad=(math.radians(80.0), 0.0, 0.0),
    )

    frame = render_to_array(tmp_path / "scene.exr")

    assert frame.shape == (52, 64)
    assert np.all(np.isfinite(frame))
    assert float(frame.min()) > 0.0
    assert float(frame.max()) > float(frame.min())


def test_scene_render_is_reproducible_from_the_same_master_seed(tmp_path) -> None:
    import numpy as np

    from spectratwin.camera.intrinsics import build_intrinsics
    from spectratwin.render.runtime import configure_scene, render_to_array, reset_scene
    from spectratwin.render.scene_builder import build_camera, build_scene
    from spectratwin.render.settings import RenderSettings

    settings = RenderSettings.for_reference_check(
        width_px=48, height_px=48, seed=0, diffuse_bounces=1
    )
    intrinsics = build_intrinsics(width_px=48, height_px=48, hfov_deg=45.0)

    def render(name: str) -> np.ndarray:
        reset_scene()
        configure_scene(settings)
        build_scene(scene=_tiny_scene(), ambient_temperature_k=293.15, master_seed=0)
        build_camera(
            intrinsics,
            location_m=(0.0, -25.0, 3.0),
            rotation_rad=(math.radians(80.0), 0.0, 0.0),
        )
        return render_to_array(tmp_path / f"{name}.exr")

    assert np.array_equal(render("first"), render("second"))


def test_camera_horizontal_field_of_view_matches_the_intrinsics() -> None:
    import bpy  # type: ignore[import-not-found]

    from spectratwin.camera.intrinsics import build_intrinsics
    from spectratwin.render.runtime import reset_scene
    from spectratwin.render.scene_builder import build_camera

    reset_scene()
    intrinsics = build_intrinsics(width_px=640, height_px=512, hfov_deg=45.0)
    build_camera(intrinsics, location_m=(0.0, 0.0, 5.0), rotation_rad=(0.0, 0.0, 0.0))

    camera = bpy.context.scene.camera.data
    assert camera.sensor_fit == "HORIZONTAL"
    assert math.degrees(camera.angle_x) == pytest.approx(45.0, rel=1e-6)


def test_scene_objects_carry_deterministic_annotation_identity() -> None:
    import bpy  # type: ignore[import-not-found]

    from spectratwin.render.runtime import reset_scene
    from spectratwin.render.scene_builder import (
        CATEGORY_PROPERTY,
        INSTANCE_INDEX_PROPERTY,
        build_scene,
    )

    scene = _tiny_scene()
    reset_scene()
    build_scene(scene=scene, ambient_temperature_k=293.15, master_seed=0)

    labelled = [obj for obj in bpy.context.scene.objects if INSTANCE_INDEX_PROPERTY in obj]
    labelled.sort(key=lambda obj: int(obj[INSTANCE_INDEX_PROPERTY]))
    assert [int(obj[INSTANCE_INDEX_PROPERTY]) for obj in labelled] == list(
        range(len(scene.objects))
    )
    assert [str(obj[CATEGORY_PROPERTY]) for obj in labelled] == [
        placed.category for placed in scene.objects
    ]
    ground = bpy.data.objects["ground"]
    assert INSTANCE_INDEX_PROPERTY not in ground
    assert CATEGORY_PROPERTY not in ground
