"""Deterministic Cycles configuration and EXR readback (SPEC-007).

This module imports ``bpy`` and therefore runs only inside Blender, launched by
``blenderproc run``. Nothing in ``scene``, ``camera``, ``thermal`` or ``sensor``
may import it (``docs/ARCHITECTURE.md``, "Dependency direction").

Readback goes through Blender's own image loader rather than a third-party
OpenEXR binding, so the render path adds no dependency that would then have to
be installed into Blender's embedded Python.
"""

from __future__ import annotations

from pathlib import Path

import bpy  # type: ignore[import-not-found]
import numpy as np

from spectratwin.render.settings import RenderSettings


def reset_scene() -> None:
    """Delete all objects and orphaned data so each render starts clean."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for datablock in list(collection):
            if datablock.users == 0:
                collection.remove(datablock)


def configure_scene(settings: RenderSettings) -> None:
    """Apply every setting that affects a pixel value.

    Each assignment below exists because its Blender default is wrong for
    radiometry, not because it is tidy to be explicit.
    """
    scene = bpy.context.scene

    scene.render.engine = "CYCLES"
    scene.cycles.device = settings.device
    scene.cycles.samples = settings.samples
    scene.cycles.seed = settings.seed

    # Denoising is a neural post-process; it is not deterministic across
    # versions and it invents values that were never rendered.
    scene.cycles.use_denoising = False

    scene.cycles.max_bounces = settings.max_bounces
    scene.cycles.diffuse_bounces = settings.diffuse_bounces
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 0

    # Clamping silently truncates high-radiance pixels. LWIR band radiance is
    # routinely in the tens or hundreds of W/sr/m^2, so any clamp would destroy
    # the signal. 0.0 disables it.
    scene.cycles.sample_clamp_direct = 0.0
    scene.cycles.sample_clamp_indirect = 0.0

    # A filmic or AGX view transform is a tone curve. It would make the render
    # look like a photograph and stop it being a measurement.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    scene.render.resolution_x = settings.width_px
    scene.render.resolution_y = settings.height_px
    scene.render.resolution_percentage = 100
    scene.render.filter_size = settings.filter_width_px
    scene.render.film_transparent = False
    scene.render.dither_intensity = 0.0

    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "NONE"


def render_to_array(output_path: Path) -> np.ndarray:
    """Render the current scene and return its linear radiance channel.

    Returns a ``(height, width)`` float array read from the red channel. The
    thermal material mapping drives all three colour channels with the same
    value, so any one of them carries the radiance; the array is flipped
    vertically because Blender stores image rows bottom-up.
    """
    scene = bpy.context.scene
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)

    image = bpy.data.images.load(str(output_path))
    try:
        width, height = image.size
        pixels = np.array(image.pixels[:], dtype=np.float32)
        rgba = pixels.reshape((height, width, 4))
        return np.flipud(rgba[:, :, 0]).copy()
    finally:
        bpy.data.images.remove(image)
