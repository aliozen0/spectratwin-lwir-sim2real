"""Controlled verification scene (SPEC-007 acceptance criteria).

A single flat patch filling the camera view, lit only by a uniform world. That
geometry is chosen so the render reduces analytically: a Lambertian surface
under a uniform hemisphere of radiance ``L`` reflects exactly ``albedo * L``, so
the rendered pixel is ``eps * B_band(To) + (1 - eps) * B_band(Tr)`` with no
geometric factor left over. That is precisely the reference bracket, which is
what makes an exact numerical comparison possible rather than a visual one.

Imports ``bpy``; runs only under ``blenderproc run``.
"""

from __future__ import annotations

from pathlib import Path

import bpy  # type: ignore[import-not-found]
import numpy as np

from spectratwin.render.nodes import build_thermal_material, set_world_radiance
from spectratwin.render.parameters import ThermalSurfaceParameters
from spectratwin.render.runtime import configure_scene, render_to_array, reset_scene
from spectratwin.render.settings import RenderSettings

#: Patch extent and camera distance, in metres. The patch is large relative to
#: the camera distance so its centre pixel sees only the patch.
PATCH_SIZE_M = 200.0
CAMERA_DISTANCE_M = 5.0


def render_controlled_patch(
    *,
    parameters: ThermalSurfaceParameters,
    world_radiance_w_sr_m2: float,
    settings: RenderSettings,
    output_path: Path,
) -> np.ndarray:
    """Render one uniform thermal patch and return its linear radiance frame.

    The world is given as radiance rather than as a temperature so that a
    caller can set it to exactly ``0.0``. That is the only way to isolate the
    emissive term: Cycles samples the world as a direct light source at the
    first hit, and that sampling is not a bounce, so ``diffuse_bounces=0`` does
    not suppress it. Measured on Blender 4.2.1, bounce counts of 0, 1 and 4 all
    returned an identical pixel. Use
    :func:`spectratwin.render.parameters.blackbody_band_radiance` to convert a
    reflected temperature for the physical case.
    """
    if world_radiance_w_sr_m2 < 0:
        raise ValueError(
            f"world_radiance_w_sr_m2 must be non-negative, got {world_radiance_w_sr_m2}"
        )

    reset_scene()
    configure_scene(settings)
    set_world_radiance(world_radiance_w_sr_m2)

    bpy.ops.mesh.primitive_plane_add(size=PATCH_SIZE_M, location=(0.0, 0.0, 0.0))
    patch = bpy.context.active_object
    patch.data.materials.append(
        build_thermal_material(f"thermal_{parameters.material_name}", parameters)
    )

    bpy.ops.object.camera_add(location=(0.0, 0.0, CAMERA_DISTANCE_M), rotation=(0.0, 0.0, 0.0))
    bpy.context.scene.camera = bpy.context.active_object

    return render_to_array(output_path)
