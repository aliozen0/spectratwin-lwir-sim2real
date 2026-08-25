"""Realise a SPEC-003 ``SceneDescription`` in Blender (SPEC-007).

SPEC-003 states its output is "a ``SceneDescription`` independent enough to
inspect/test before rendering, plus Blender scene realization at adapter layer".
This module is that adapter layer, and it is the only place the two meet: scene
sampling stays renderer-free and this module adds no sampling of its own beyond
the seeded thermal material draws.

Geometry is placeholder primitives sized from each asset's recorded footprint.
Licensed meshes are Roadmap Session 6; using primitives here keeps the thermal
path testable without importing an asset whose licence has not been reviewed.

Imports ``bpy``; runs only under ``blenderproc run``.
"""

from __future__ import annotations

import math
from typing import Any

import bpy  # type: ignore[import-not-found]

from spectratwin.camera.intrinsics import CameraIntrinsics
from spectratwin.randomness.seed import new_generator
from spectratwin.render.nodes import build_thermal_material, set_world_radiance
from spectratwin.render.parameters import blackbody_band_radiance, derive_surface_parameters
from spectratwin.scene.placement import PlacedObject
from spectratwin.scene.scene import SceneDescription
from spectratwin.thermal.materials import get_thermal_material

#: Which thermal class each project category is rendered with.
# ponytail: one thermal region per object; docs/THERMAL_MODEL.md calls for
# vehicle body/tyre/engine and person body/clothing regions, which is Roadmap
# Session 13. Split this mapping into regions when that lands.
CATEGORY_MATERIAL: dict[str, str] = {
    "car": "painted_metal",
    "person": "fabric",
    "bicycle": "rubber",
}

#: Placeholder object heights in metres, by category. Footprint length and width
#: come from the asset registry; height is not recorded there, so these are
#: stated here rather than silently defaulted.
CATEGORY_HEIGHT_M: dict[str, float] = {"car": 1.5, "person": 1.7, "bicycle": 1.1}

#: The ground plane's thermal class and its extent in metres.
GROUND_MATERIAL = "asphalt"
GROUND_SIZE_M = 400.0

#: How far below ambient the reflected sky is assumed to sit, in kelvin. A
#: uniform cold background; docs/THERMAL_MODEL.md defers any sky gradient, and
#: SPEC-007 records it as an open question.
SKY_OFFSET_K = 20.0


def _add_object_primitive(placed: PlacedObject) -> Any:
    """Add one box sized from the placed object's recorded footprint."""
    height_m = CATEGORY_HEIGHT_M[placed.category]
    x_m, y_m = placed.position_m

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x_m, y_m, height_m / 2.0))
    primitive = bpy.context.active_object
    primitive.scale = (placed.footprint_length_m, placed.footprint_width_m, height_m)
    primitive.rotation_euler = (0.0, 0.0, placed.orientation_rad)
    primitive.name = f"{placed.category}_{placed.asset_id}"
    return primitive


def build_scene(*, scene: SceneDescription, ambient_temperature_k: float, master_seed: int) -> None:
    """Realise ``scene`` as primitives with seeded thermal materials.

    Material draws use ``derive_subseed`` through ``new_generator`` with a label
    path that includes the scene's own sample seed, so the same
    ``SceneDescription`` and ``master_seed`` always draw the same materials
    (``docs/REPRODUCIBILITY.md``, "Randomness").
    """
    if ambient_temperature_k <= 0:
        raise ValueError(
            f"ambient_temperature_k must be positive (absolute), got {ambient_temperature_k}"
        )

    rng = new_generator(master_seed, "thermal", "scene", str(scene.sample_seed))

    sky_temperature_k = ambient_temperature_k - SKY_OFFSET_K
    set_world_radiance(blackbody_band_radiance(temperature_k=sky_temperature_k))

    ground_surface = get_thermal_material(GROUND_MATERIAL).draw(
        rng, ambient_temperature_k=ambient_temperature_k
    )
    bpy.ops.mesh.primitive_plane_add(size=GROUND_SIZE_M, location=(0.0, 0.0, 0.0))
    ground = bpy.context.active_object
    ground.name = "ground"
    ground.data.materials.append(
        build_thermal_material("thermal_ground", derive_surface_parameters(surface=ground_surface))
    )

    for index, placed in enumerate(scene.objects):
        material_name = CATEGORY_MATERIAL[placed.category]
        surface = get_thermal_material(material_name).draw(
            rng, ambient_temperature_k=ambient_temperature_k
        )
        primitive = _add_object_primitive(placed)
        primitive.data.materials.append(
            build_thermal_material(
                f"thermal_{placed.category}_{index}",
                derive_surface_parameters(surface=surface),
            )
        )


def build_camera(
    intrinsics: CameraIntrinsics,
    *,
    location_m: tuple[float, float, float],
    rotation_rad: tuple[float, float, float],
) -> None:
    """Create a Blender camera matching ``intrinsics``.

    Blender expresses a pinhole camera by sensor size and focal length, or
    directly by field of view. The field-of-view form is used here so the
    conversion cannot drift from SPEC-004's own
    ``focal_px = (width_px / 2) / tan(hfov / 2)`` relation: horizontal sensor fit
    plus ``angle_x`` reproduces exactly the same pinhole.
    """
    bpy.ops.object.camera_add(location=location_m, rotation=rotation_rad)
    camera_object = bpy.context.active_object
    camera = camera_object.data
    camera.sensor_fit = "HORIZONTAL"
    camera.angle_x = math.radians(intrinsics.hfov_deg)
    bpy.context.scene.render.resolution_x = intrinsics.width_px
    bpy.context.scene.render.resolution_y = intrinsics.height_px
    bpy.context.scene.camera = camera_object
