"""SPEC-005 renderer regressions for instance IDs, depth and state restoration."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = pytest.mark.renderer


def _camera(*, height_m: float = 10.0) -> None:
    import bpy  # type: ignore[import-not-found]

    bpy.ops.object.camera_add(location=(0.0, 0.0, height_m), rotation=(0.0, 0.0, 0.0))
    camera = bpy.context.active_object
    camera.data.angle_x = math.radians(45.0)
    bpy.context.scene.camera = camera


def _emissive_plane(
    *,
    name: str,
    strength: float,
    size_m: float,
    location_m: tuple[float, float, float],
    instance_index: int | None,
) -> object:
    import bpy  # type: ignore[import-not-found]

    from spectratwin.render.scene_builder import (
        CATEGORY_PROPERTY,
        INSTANCE_INDEX_PROPERTY,
    )

    bpy.ops.mesh.primitive_plane_add(size=size_m, location=location_m)
    plane = bpy.context.active_object
    plane.name = name
    material = bpy.data.materials.new(name=f"{name}_thermal")
    material.use_nodes = True
    material.node_tree.nodes.clear()
    emission = material.node_tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = strength
    output = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    plane.data.materials.append(material)
    if instance_index is not None:
        plane[INSTANCE_INDEX_PROPERTY] = instance_index
        plane[CATEGORY_PROPERTY] = "person" if instance_index == 0 else "car"
    return plane


def _configure(width: int = 32, height: int = 32) -> None:
    from spectratwin.render.runtime import configure_scene, reset_scene
    from spectratwin.render.settings import RenderSettings

    reset_scene()
    configure_scene(
        RenderSettings.for_reference_check(
            width_px=width,
            height_px=height,
            seed=17,
            diffuse_bounces=0,
        )
    )


def _geometry_state_snapshot() -> tuple[object, ...]:
    import bpy  # type: ignore[import-not-found]

    scene = bpy.context.scene
    node_tree = scene.node_tree
    node_state = (
        ()
        if node_tree is None
        else tuple((node.as_pointer(), bool(node.mute)) for node in node_tree.nodes)
    )
    object_data = tuple(
        sorted(
            (obj.as_pointer(), obj.data.as_pointer()) for obj in scene.objects if obj.type == "MESH"
        )
    )
    return (
        scene.world.as_pointer() if scene.world is not None else None,
        bool(scene.use_nodes),
        node_tree.as_pointer() if node_tree is not None else None,
        node_state,
        object_data,
        bool(bpy.context.view_layer.use_pass_z),
        scene.cycles.samples,
        bool(scene.cycles.use_adaptive_sampling),
        bool(scene.cycles.use_denoising),
        scene.cycles.filter_width,
        scene.cycles.seed,
        bool(scene.render.film_transparent),
        scene.render.dither_intensity,
        bool(scene.render.use_motion_blur),
        bool(scene.render.use_compositing),
        scene.view_settings.view_transform,
        scene.view_settings.look,
        scene.view_settings.exposure,
        scene.view_settings.gamma,
    )


@pytest.mark.parametrize(
    "pixels",
    [
        np.array([[0.5]], dtype=np.float32),
        np.array([[-1.0]], dtype=np.float32),
        np.array([[np.nan]], dtype=np.float32),
        np.zeros((1, 1, 1), dtype=np.float32),
    ],
)
def test_instance_decode_rejects_fractional_negative_nonfinite_or_non_2d_values(
    pixels: np.ndarray,
) -> None:
    from spectratwin.render.geometry_pass import GeometryPassDecodeError, decode_instance_ids

    with pytest.raises(GeometryPassDecodeError):
        decode_instance_ids(pixels, max_instance_id=2)


def test_geometry_pass_returns_exact_ids_background_and_two_instances(tmp_path) -> None:
    from spectratwin.render.geometry_pass import render_geometry_pass

    _configure()
    _emissive_plane(
        name="left",
        strength=41.0,
        size_m=2.0,
        location_m=(-1.8, 0.0, 0.0),
        instance_index=0,
    )
    _emissive_plane(
        name="right",
        strength=73.0,
        size_m=2.0,
        location_m=(1.8, 0.0, 0.0),
        instance_index=1,
    )
    _camera()

    result = render_geometry_pass(
        instance_exr_path=tmp_path / "instances.exr",
        depth_exr_path=tmp_path / "depth.exr",
    )

    assert result.instance_id_map.dtype == np.uint32
    assert set(np.unique(result.instance_id_map)) == {0, 1, 2}
    assert result.depth_m.shape == result.instance_id_map.shape == (32, 32)
    assert (tmp_path / "instances.exr").is_file()
    assert (tmp_path / "depth.exr").is_file()


def test_geometry_pass_rejects_noncontiguous_instance_indices(tmp_path) -> None:
    from spectratwin.render.geometry_pass import GeometryPassSceneError, render_geometry_pass

    _configure()
    _emissive_plane(
        name="gap",
        strength=41.0,
        size_m=2.0,
        location_m=(0.0, 0.0, 0.0),
        instance_index=1,
    )
    _camera()

    with pytest.raises(GeometryPassSceneError, match="contiguous"):
        render_geometry_pass(
            instance_exr_path=tmp_path / "instances.exr",
            depth_exr_path=tmp_path / "depth.exr",
        )


def test_known_plane_depth_and_thermal_render_are_preserved_bit_exactly(tmp_path) -> None:
    from spectratwin.render.geometry_pass import render_geometry_pass
    from spectratwin.render.runtime import render_to_array

    _configure()
    _emissive_plane(
        name="full_frame",
        strength=54.6553,
        size_m=100.0,
        location_m=(0.0, 0.0, 0.0),
        instance_index=0,
    )
    _camera(height_m=10.0)
    before = render_to_array(tmp_path / "thermal_before.exr")
    state_before = _geometry_state_snapshot()

    geometry = render_geometry_pass(
        instance_exr_path=tmp_path / "instances.exr",
        depth_exr_path=tmp_path / "depth.exr",
    )
    state_after = _geometry_state_snapshot()
    after = render_to_array(tmp_path / "thermal_after.exr")

    centre = geometry.depth_m[geometry.depth_m.shape[0] // 2, geometry.depth_m.shape[1] // 2]
    assert centre == pytest.approx(10.0, rel=1e-4)
    assert np.all(geometry.instance_id_map == 1)
    assert state_after == state_before
    assert np.array_equal(before, after)


def test_partial_occlusion_reduces_visible_area_and_changes_tight_bbox(tmp_path) -> None:
    from spectratwin.annotation.masks import extract_visible_instances
    from spectratwin.render.geometry_pass import render_geometry_pass

    _configure(width=48, height=48)
    _emissive_plane(
        name="target",
        strength=50.0,
        size_m=4.0,
        location_m=(0.0, 0.0, 0.0),
        instance_index=0,
    )
    _camera()
    unoccluded = render_geometry_pass(
        instance_exr_path=tmp_path / "unoccluded_instances.exr",
        depth_exr_path=tmp_path / "unoccluded_depth.exr",
    )
    unoccluded_target = extract_visible_instances(unoccluded.instance_id_map)[0]

    _emissive_plane(
        name="occluder",
        strength=80.0,
        size_m=4.0,
        location_m=(1.8, 0.0, 1.0),
        instance_index=1,
    )
    occluded = render_geometry_pass(
        instance_exr_path=tmp_path / "occluded_instances.exr",
        depth_exr_path=tmp_path / "occluded_depth.exr",
    )
    occluded_by_index = {
        instance.instance_index: instance
        for instance in extract_visible_instances(occluded.instance_id_map)
    }
    occluded_target = occluded_by_index[0]

    assert occluded_target.visible_area_px < unoccluded_target.visible_area_px
    assert occluded_target.bbox_xywh != unoccluded_target.bbox_xywh


def test_decode_failure_still_restores_the_thermal_scene(tmp_path, monkeypatch) -> None:
    import spectratwin.render.geometry_pass as geometry_pass
    from spectratwin.render.runtime import render_to_array

    _configure(width=16, height=16)
    _emissive_plane(
        name="failure_probe",
        strength=54.6553,
        size_m=100.0,
        location_m=(0.0, 0.0, 0.0),
        instance_index=0,
    )
    _camera()
    before = render_to_array(tmp_path / "failure_before.exr")
    state_before = _geometry_state_snapshot()
    monkeypatch.setattr(
        geometry_pass,
        "_read_exr_red_channel",
        lambda _path: np.full((16, 16), 0.5, dtype=np.float32),
    )

    with pytest.raises(geometry_pass.GeometryPassDecodeError):
        geometry_pass.render_geometry_pass(
            instance_exr_path=tmp_path / "failed_instances.exr",
            depth_exr_path=tmp_path / "failed_depth.exr",
        )

    state_after = _geometry_state_snapshot()
    after = render_to_array(tmp_path / "failure_after.exr")
    assert state_after == state_before
    assert np.array_equal(before, after)


@pytest.mark.parametrize("invalid_depth", [np.nan, np.inf, -1.0, 0.0])
def test_invalid_depth_still_restores_the_thermal_scene(
    tmp_path, monkeypatch, invalid_depth: float
) -> None:
    import spectratwin.render.geometry_pass as geometry_pass
    from spectratwin.render.runtime import render_to_array

    _configure(width=16, height=16)
    _emissive_plane(
        name="depth_failure_probe",
        strength=54.6553,
        size_m=100.0,
        location_m=(0.0, 0.0, 0.0),
        instance_index=0,
    )
    _camera()
    before = render_to_array(tmp_path / "depth_failure_before.exr")
    state_before = _geometry_state_snapshot()
    channels = iter(
        (
            np.ones((16, 16), dtype=np.float32),
            np.full((16, 16), invalid_depth, dtype=np.float32),
        )
    )
    monkeypatch.setattr(geometry_pass, "_read_exr_red_channel", lambda _path: next(channels))

    with pytest.raises(geometry_pass.GeometryPassDecodeError, match="depth"):
        geometry_pass.render_geometry_pass(
            instance_exr_path=tmp_path / "invalid_depth_instances.exr",
            depth_exr_path=tmp_path / "invalid_depth.exr",
        )

    state_after = _geometry_state_snapshot()
    after = render_to_array(tmp_path / "depth_failure_after.exr")
    assert state_after == state_before
    assert np.array_equal(before, after)
