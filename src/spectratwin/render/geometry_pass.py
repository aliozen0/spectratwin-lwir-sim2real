"""Dedicated instance-ID and optical-axis depth render pass (SPEC-005).

Imports ``bpy`` and runs only under ``blenderproc run``. The pass temporarily
replaces mesh data/materials and renderer/compositor state, then restores the
SPEC-007 thermal scene before returning.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bpy  # type: ignore[import-not-found]
import numpy as np

from spectratwin.real_data.taxonomy import PROJECT_CATEGORIES
from spectratwin.render.scene_builder import CATEGORY_PROPERTY, INSTANCE_INDEX_PROPERTY

INSTANCE_ID_TOLERANCE = 1e-4


class GeometryPassError(RuntimeError):
    """Base failure for the Blender-bound geometry pass."""


class GeometryPassDecodeError(GeometryPassError):
    """Raised when float EXR pixels are not exact-enough non-negative IDs."""


class GeometryPassSceneError(GeometryPassError):
    """Raised when realised Blender objects violate annotation identity."""


@dataclass(frozen=True, slots=True)
class GeometryPassResult:
    """Decoded geometry products from one dedicated render."""

    instance_id_map: np.ndarray
    depth_m: np.ndarray


@dataclass(frozen=True, slots=True)
class _AttributeSnapshot:
    owner: Any
    name: str
    value: Any


def decode_instance_ids(
    pixels: np.ndarray,
    *,
    max_instance_id: int,
    tolerance: float = INSTANCE_ID_TOLERANCE,
) -> np.ndarray:
    """Decode float EXR values only when each is an in-range integer."""
    array = np.asarray(pixels)
    if array.ndim != 2:
        raise GeometryPassDecodeError(f"instance EXR must decode to 2D, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise GeometryPassDecodeError("instance EXR contains non-finite values")
    if np.any(array < 0.0):
        raise GeometryPassDecodeError("instance EXR contains negative values")
    rounded = np.rint(array)
    max_error = float(np.max(np.abs(array - rounded))) if array.size else 0.0
    if max_error > tolerance:
        raise GeometryPassDecodeError(
            f"instance EXR contains fractional IDs (max error {max_error:.6g})"
        )
    if np.any(rounded > max_instance_id):
        raise GeometryPassDecodeError(
            f"instance EXR contains ID above declared maximum {max_instance_id}"
        )
    if max_instance_id > np.iinfo(np.uint32).max:
        raise GeometryPassDecodeError("declared maximum instance ID exceeds uint32")
    return rounded.astype(np.uint32)


def _read_exr_red_channel(path: Path) -> np.ndarray:
    if not path.is_file():
        raise GeometryPassError(f"geometry pass did not produce expected {path.name}")
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = image.size
        pixels = np.array(image.pixels[:], dtype=np.float32)
        rgba = pixels.reshape((height, width, 4))
        return np.flipud(rgba[:, :, 0]).copy()
    finally:
        bpy.data.images.remove(image)


def _set_temporarily(
    snapshots: list[_AttributeSnapshot], owner: Any, name: str, value: Any
) -> None:
    if hasattr(owner, name):
        snapshots.append(_AttributeSnapshot(owner=owner, name=name, value=getattr(owner, name)))
        setattr(owner, name, value)


def _emission_material(*, name: str, strength: int) -> Any:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    if hasattr(material, "cycles"):
        material.cycles.sample_as_light = False
    nodes = material.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = float(strength)
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _labelled_meshes(scene: Any) -> tuple[list[Any], int]:
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    seen_indices: set[int] = set()
    max_instance_id = 0
    for obj in meshes:
        if INSTANCE_INDEX_PROPERTY not in obj:
            continue
        raw_index = obj[INSTANCE_INDEX_PROPERTY]
        raw_category = obj.get(CATEGORY_PROPERTY)
        if type(raw_index) is not int or raw_index < 0:
            raise GeometryPassSceneError(f"object {obj.name!r} has invalid instance index")
        if raw_index in seen_indices:
            raise GeometryPassSceneError(f"duplicate instance index {raw_index}")
        if raw_category not in PROJECT_CATEGORIES:
            raise GeometryPassSceneError(f"object {obj.name!r} has invalid project category")
        seen_indices.add(raw_index)
        max_instance_id = max(max_instance_id, raw_index + 1)
    if seen_indices != set(range(len(seen_indices))):
        raise GeometryPassSceneError("labelled instance indices must be contiguous from zero")
    return meshes, max_instance_id


def _one_output_path(directory: Path, prefix: str) -> Path:
    paths = sorted(directory.glob(f"{prefix}*.exr"))
    if len(paths) != 1:
        raise GeometryPassError(f"geometry pass expected one {prefix} EXR, found {len(paths)}")
    return paths[0]


def render_geometry_pass(
    *,
    instance_exr_path: Path,
    depth_exr_path: Path,
) -> GeometryPassResult:
    """Render scalar instance IDs and Z depth, restoring thermal state exactly."""
    if instance_exr_path == depth_exr_path:
        raise GeometryPassError("instance and depth output paths must differ")
    instance_exr_path.parent.mkdir(parents=True, exist_ok=True)
    depth_exr_path.parent.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    mesh_objects, max_instance_id = _labelled_meshes(scene)
    attribute_snapshots: list[_AttributeSnapshot] = []
    object_data_snapshots: list[tuple[Any, Any]] = []
    temporary_meshes: list[Any] = []
    temporary_materials: list[Any] = []
    temporary_nodes: list[Any] = []
    output_mute_snapshots: list[tuple[Any, bool]] = []
    original_world = scene.world
    temporary_world: Any | None = None

    with tempfile.TemporaryDirectory(
        prefix="spectratwin-geometry-", dir=instance_exr_path.parent
    ) as raw:
        temporary_directory = Path(raw)
        try:
            _set_temporarily(attribute_snapshots, scene.cycles, "samples", 1)
            _set_temporarily(attribute_snapshots, scene.cycles, "use_adaptive_sampling", False)
            _set_temporarily(attribute_snapshots, scene.cycles, "use_denoising", False)
            _set_temporarily(attribute_snapshots, scene.cycles, "filter_width", 0.0)
            _set_temporarily(attribute_snapshots, scene.cycles, "seed", 0)
            _set_temporarily(attribute_snapshots, scene.render, "film_transparent", False)
            _set_temporarily(attribute_snapshots, scene.render, "dither_intensity", 0.0)
            _set_temporarily(attribute_snapshots, scene.render, "use_motion_blur", False)
            _set_temporarily(attribute_snapshots, scene.view_settings, "view_transform", "Standard")
            _set_temporarily(attribute_snapshots, scene.view_settings, "look", "None")
            _set_temporarily(attribute_snapshots, scene.view_settings, "exposure", 0.0)
            _set_temporarily(attribute_snapshots, scene.view_settings, "gamma", 1.0)
            _set_temporarily(attribute_snapshots, view_layer, "use_pass_z", True)
            _set_temporarily(attribute_snapshots, scene.render, "use_compositing", True)
            _set_temporarily(attribute_snapshots, scene, "use_nodes", True)

            new_world = bpy.data.worlds.new("spectratwin_geometry_world")
            temporary_world = new_world
            new_world.use_nodes = True
            world_tree = new_world.node_tree
            if world_tree is None:
                raise GeometryPassSceneError("temporary world did not create a node tree")
            background = world_tree.nodes.get("Background")
            if background is None:
                raise GeometryPassSceneError("temporary world has no Background node")
            background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            background.inputs["Strength"].default_value = 0.0
            scene.world = temporary_world

            materials_by_id: dict[int, Any] = {}
            for obj in mesh_objects:
                instance_id = (
                    int(obj[INSTANCE_INDEX_PROPERTY]) + 1 if INSTANCE_INDEX_PROPERTY in obj else 0
                )
                material = materials_by_id.get(instance_id)
                if material is None:
                    material = _emission_material(
                        name=f"spectratwin_geometry_id_{instance_id}", strength=instance_id
                    )
                    materials_by_id[instance_id] = material
                    temporary_materials.append(material)

                original_data = obj.data
                temporary_data = original_data.copy()
                object_data_snapshots.append((obj, original_data))
                temporary_meshes.append(temporary_data)
                obj.data = temporary_data
                temporary_data.materials.clear()
                temporary_data.materials.append(material)
                for polygon in temporary_data.polygons:
                    polygon.material_index = 0

            node_tree = scene.node_tree
            for node in node_tree.nodes:
                if node.bl_idname == "CompositorNodeOutputFile":
                    output_mute_snapshots.append((node, bool(node.mute)))
                    node.mute = True

            render_layers = node_tree.nodes.new("CompositorNodeRLayers")
            instance_output = node_tree.nodes.new("CompositorNodeOutputFile")
            depth_combine = node_tree.nodes.new("CompositorNodeCombineColor")
            depth_output = node_tree.nodes.new("CompositorNodeOutputFile")
            temporary_nodes.extend((render_layers, instance_output, depth_combine, depth_output))

            for output_node, prefix in (
                (instance_output, "instance_"),
                (depth_output, "depth_"),
            ):
                output_node.base_path = str(temporary_directory)
                output_node.file_slots[0].path = prefix
                output_node.format.file_format = "OPEN_EXR"
                output_node.format.color_mode = "RGB"
                output_node.format.color_depth = "32"
                output_node.format.exr_codec = "NONE"

            depth_combine.mode = "HSV"
            node_tree.links.new(render_layers.outputs["Image"], instance_output.inputs["Image"])
            node_tree.links.new(render_layers.outputs["Depth"], depth_combine.inputs[2])
            node_tree.links.new(depth_combine.outputs["Image"], depth_output.inputs["Image"])

            bpy.ops.render.render()
            temporary_instance = _one_output_path(temporary_directory, "instance_")
            temporary_depth = _one_output_path(temporary_directory, "depth_")
            raw_instance = _read_exr_red_channel(temporary_instance)
            depth_m = _read_exr_red_channel(temporary_depth)
            instance_id_map = decode_instance_ids(
                raw_instance,
                max_instance_id=max_instance_id,
            )
            if depth_m.ndim != 2 or depth_m.shape != instance_id_map.shape:
                raise GeometryPassDecodeError(
                    f"depth shape {depth_m.shape} differs from instances {instance_id_map.shape}"
                )
            if not np.all(np.isfinite(depth_m)) or np.any(depth_m <= 0.0):
                raise GeometryPassDecodeError(
                    "depth EXR contains non-finite or non-positive values"
                )
        finally:
            if scene.node_tree is not None:
                for node in reversed(temporary_nodes):
                    if node in scene.node_tree.nodes.values():
                        scene.node_tree.nodes.remove(node)
            for node, mute in output_mute_snapshots:
                node.mute = mute
            for obj, original_data in reversed(object_data_snapshots):
                obj.data = original_data
            for mesh in reversed(temporary_meshes):
                if mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
            scene.world = original_world
            if temporary_world is not None and temporary_world.users == 0:
                bpy.data.worlds.remove(temporary_world)
            for material in reversed(temporary_materials):
                if material.users == 0:
                    bpy.data.materials.remove(material)
            for snapshot in reversed(attribute_snapshots):
                setattr(snapshot.owner, snapshot.name, snapshot.value)

        os.replace(temporary_instance, instance_exr_path)
        os.replace(temporary_depth, depth_exr_path)

    return GeometryPassResult(instance_id_map=instance_id_map, depth_m=depth_m)
