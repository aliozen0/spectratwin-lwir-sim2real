"""Renderer configuration proof: what goes into an Emission node must come back
out of the EXR. Runs only under ``blenderproc run``; see the module docstring of
``spectratwin.render.runtime`` for why it cannot run under plain pytest."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.renderer


def _emissive_plane(strength: float, name: str) -> None:
    import bpy  # type: ignore[import-not-found]

    bpy.ops.mesh.primitive_plane_add(size=100.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    material.node_tree.nodes.clear()
    emission = material.node_tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = strength
    output = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    plane.data.materials.append(material)

    bpy.ops.object.camera_add(location=(0.0, 0.0, 10.0), rotation=(0.0, 0.0, 0.0))
    bpy.context.scene.camera = bpy.context.active_object


def test_emission_radiance_survives_the_render_pipeline(tmp_path) -> None:
    from spectratwin.render.runtime import configure_scene, render_to_array, reset_scene
    from spectratwin.render.settings import RenderSettings

    expected_radiance = 54.6553

    reset_scene()
    configure_scene(
        RenderSettings.for_reference_check(width_px=32, height_px=32, seed=0, diffuse_bounces=0)
    )
    _emissive_plane(expected_radiance, "probe")

    frame = render_to_array(tmp_path / "probe.exr")

    centre = frame[frame.shape[0] // 2, frame.shape[1] // 2]
    assert centre == pytest.approx(expected_radiance, rel=1e-4)


def test_a_value_far_above_one_is_not_clamped(tmp_path) -> None:
    """Light clamping and tone mapping both truncate high values. LWIR band
    radiance is routinely far above 1.0, so this must survive."""
    from spectratwin.render.runtime import configure_scene, render_to_array, reset_scene
    from spectratwin.render.settings import RenderSettings

    expected_radiance = 5000.0

    reset_scene()
    configure_scene(
        RenderSettings.for_reference_check(width_px=16, height_px=16, seed=0, diffuse_bounces=0)
    )
    _emissive_plane(expected_radiance, "bright")

    frame = render_to_array(tmp_path / "bright.exr")
    centre = frame[frame.shape[0] // 2, frame.shape[1] // 2]
    assert centre == pytest.approx(expected_radiance, rel=1e-4)
