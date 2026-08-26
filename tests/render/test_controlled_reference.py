"""SPEC-007 acceptance: the rendered controlled patch must equal the SPEC-006
CPU reference numerically, not merely look plausible."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.renderer

TOLERANCE = 2e-3


def _parameters(emissivity: float, temperature_k: float, name: str):
    from spectratwin.render.parameters import derive_surface_parameters
    from spectratwin.thermal.materials import ResolvedThermalSurface

    return derive_surface_parameters(
        surface=ResolvedThermalSurface(
            material_name=name, emissivity=emissivity, temperature_k=temperature_k
        )
    )


def _world(temperature_k: float) -> float:
    from spectratwin.render.parameters import blackbody_band_radiance

    return blackbody_band_radiance(temperature_k=temperature_k)


def test_rendered_patch_equals_the_full_reference_bracket(tmp_path) -> None:
    """SPEC-007 acceptance criterion 1: with a uniform world at B_band(Tr), the
    pixel equals band_radiance(...) at transmittance 1."""
    from spectratwin.render.controlled import render_controlled_patch
    from spectratwin.render.settings import RenderSettings
    from spectratwin.thermal.radiometry import band_radiance

    emissivity, object_k, reflected_k = 0.60, 320.0, 275.0
    parameters = _parameters(emissivity, object_k, "controlled")
    settings = RenderSettings.for_reference_check(
        width_px=64, height_px=64, seed=0, diffuse_bounces=1
    )

    frame = render_controlled_patch(
        parameters=parameters,
        world_radiance_w_sr_m2=_world(reflected_k),
        settings=settings,
        output_path=tmp_path / "bracket.exr",
    )

    expected = band_radiance(
        object_temperature_k=object_k,
        reflected_temperature_k=reflected_k,
        atmospheric_temperature_k=290.0,
        emissivity=emissivity,
        transmittance=1.0,
    )
    centre = frame[frame.shape[0] // 2, frame.shape[1] // 2]
    assert centre == pytest.approx(expected, rel=TOLERANCE)


def test_a_zero_radiance_world_isolates_the_emissive_term(tmp_path) -> None:
    """SPEC-007 acceptance criterion 2: with nothing for the surface to reflect,
    the pixel equals eps * B_band(To) alone.

    A zero-radiance world is the isolating mechanism, not a bounce count. Cycles
    samples the world as a direct light source at the first hit, and that is not
    a bounce: measured on Blender 4.2.1, diffuse_bounces of 0, 1 and 4 all
    produced an identical pixel.
    """
    from spectratwin.render.controlled import render_controlled_patch
    from spectratwin.render.settings import RenderSettings

    parameters = _parameters(0.60, 320.0, "controlled")
    settings = RenderSettings.for_reference_check(
        width_px=64, height_px=64, seed=0, diffuse_bounces=1
    )

    frame = render_controlled_patch(
        parameters=parameters,
        world_radiance_w_sr_m2=0.0,
        settings=settings,
        output_path=tmp_path / "emissive.exr",
    )

    centre = frame[frame.shape[0] // 2, frame.shape[1] // 2]
    assert centre == pytest.approx(parameters.emission_radiance_w_sr_m2, rel=TOLERANCE)


def test_hot_patch_renders_brighter_than_cold_patch(tmp_path) -> None:
    """SPEC-007 acceptance criterion 3."""
    from spectratwin.render.controlled import render_controlled_patch
    from spectratwin.render.settings import RenderSettings

    settings = RenderSettings.for_reference_check(
        width_px=32, height_px=32, seed=0, diffuse_bounces=1
    )

    def render(temperature_k: float, name: str) -> float:
        frame = render_controlled_patch(
            parameters=_parameters(0.95, temperature_k, name),
            world_radiance_w_sr_m2=_world(280.0),
            settings=settings,
            output_path=tmp_path / f"{name}.exr",
        )
        return float(frame[frame.shape[0] // 2, frame.shape[1] // 2])

    assert render(330.0, "hot") > render(285.0, "cold")


def test_low_emissivity_renders_darker_against_a_cold_world(tmp_path) -> None:
    """SPEC-007 acceptance criterion 4: under a world colder than the object, a
    low-emissivity surface reflects that cold world and appears darker."""
    from spectratwin.render.controlled import render_controlled_patch
    from spectratwin.render.settings import RenderSettings

    settings = RenderSettings.for_reference_check(
        width_px=32, height_px=32, seed=0, diffuse_bounces=1
    )

    def render(emissivity: float, name: str) -> float:
        frame = render_controlled_patch(
            parameters=_parameters(emissivity, 320.0, name),
            world_radiance_w_sr_m2=_world(250.0),
            settings=settings,
            output_path=tmp_path / f"{name}.exr",
        )
        return float(frame[frame.shape[0] // 2, frame.shape[1] // 2])

    assert render(0.50, "low_emissivity") < render(0.98, "high_emissivity")


def test_visible_colour_does_not_change_thermal_output(tmp_path) -> None:
    """SPEC-007 acceptance criterion 5, and the anti-gate in QUALITY_GATES.md
    against "calling grayscale RGB thermal"."""
    import bpy  # type: ignore[import-not-found]
    import numpy as np

    from spectratwin.render.controlled import render_controlled_patch
    from spectratwin.render.settings import RenderSettings

    parameters = _parameters(0.9, 310.0, "colour_probe")
    settings = RenderSettings.for_reference_check(
        width_px=32, height_px=32, seed=0, diffuse_bounces=1
    )

    first = render_controlled_patch(
        parameters=parameters,
        world_radiance_w_sr_m2=_world(280.0),
        settings=settings,
        output_path=tmp_path / "colour_a.exr",
    )
    for material in bpy.data.materials:
        material.diffuse_color = (1.0, 0.0, 0.0, 1.0)
    second = render_controlled_patch(
        parameters=parameters,
        world_radiance_w_sr_m2=_world(280.0),
        settings=settings,
        output_path=tmp_path / "colour_b.exr",
    )

    assert np.array_equal(first, second)


def test_the_same_seed_reproduces_the_same_image(tmp_path) -> None:
    """SPEC-007 acceptance criterion 6."""
    import numpy as np

    from spectratwin.render.controlled import render_controlled_patch
    from spectratwin.render.settings import RenderSettings

    parameters = _parameters(0.8, 305.0, "repeat")
    settings = RenderSettings.for_reference_check(
        width_px=32, height_px=32, seed=42, diffuse_bounces=1
    )

    first = render_controlled_patch(
        parameters=parameters,
        world_radiance_w_sr_m2=_world(280.0),
        settings=settings,
        output_path=tmp_path / "seed_a.exr",
    )
    second = render_controlled_patch(
        parameters=parameters,
        world_radiance_w_sr_m2=_world(280.0),
        settings=settings,
        output_path=tmp_path / "seed_b.exr",
    )

    assert np.array_equal(first, second)
