"""Cycles material nodes for thermal surfaces (SPEC-007, ADR-010).

The node tree is deliberately small: an Emission node carrying
``eps * B_band(To)`` and a Diffuse BSDF carrying albedo ``1 - eps``, added
together. Cycles then evaluates
``eps * B(To) + (1 - eps) * (incident radiance)`` per shading point, which is
the reference model's bracket with the reflected term solved rather than
assumed.

Imports ``bpy``; runs only under ``blenderproc run``.
"""

from __future__ import annotations

from typing import Any

import bpy  # type: ignore[import-not-found]

from spectratwin.render.parameters import ThermalSurfaceParameters


def build_thermal_material(name: str, parameters: ThermalSurfaceParameters) -> Any:
    """Build the Emission + Diffuse node tree for one thermal surface.

    Colour inputs are white on purpose. Thermal intensity must not depend on a
    visible colour channel (SPEC-007), so all three channels carry the same
    radiance and the caller may read any one of them.
    """
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = parameters.emission_radiance_w_sr_m2

    albedo = parameters.diffuse_albedo
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (albedo, albedo, albedo, 1.0)
    # Roughness MUST stay 0.0. Cycles' Diffuse BSDF is Lambertian only at
    # roughness 0; above it the node switches to Oren-Nayar, which reflects
    # less of a uniform environment. Measured on Blender 4.2.1 with albedo 1.0
    # under a uniform world: roughness 0.0 returns exactly the world radiance
    # (ratio 1.000000), roughness 1.0 returns 0.766220 of it. Only the
    # Lambertian case satisfies the reference model's (1 - eps) * B(Tr) term.
    diffuse.inputs["Roughness"].default_value = 0.0

    add = tree.nodes.new("ShaderNodeAddShader")
    output = tree.nodes.new("ShaderNodeOutputMaterial")

    tree.links.new(emission.outputs["Emission"], add.inputs[0])
    tree.links.new(diffuse.outputs["BSDF"], add.inputs[1])
    tree.links.new(add.outputs["Shader"], output.inputs["Surface"])

    return material


def set_world_radiance(radiance_w_sr_m2: float) -> None:
    """Set the uniform world background to ``B_band(Tr)``.

    This is what the Diffuse term reflects. Making it uniform is what lets the
    controlled scene reduce analytically to the reference model.
    """
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("thermal_world")
        bpy.context.scene.world = world

    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()

    background = tree.nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs["Strength"].default_value = radiance_w_sr_m2

    output = tree.nodes.new("ShaderNodeOutputWorld")
    tree.links.new(background.outputs["Background"], output.inputs["Surface"])
