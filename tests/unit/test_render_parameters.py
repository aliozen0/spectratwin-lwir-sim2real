import pytest
from pydantic import ValidationError

from spectratwin.render.parameters import (
    ThermalSurfaceParameters,
    blackbody_band_radiance,
    derive_surface_parameters,
)
from spectratwin.thermal.materials import ResolvedThermalSurface
from spectratwin.thermal.radiometry import band_radiance


def _surface(emissivity: float, temperature_k: float) -> ResolvedThermalSurface:
    return ResolvedThermalSurface(
        material_name="test", emissivity=emissivity, temperature_k=temperature_k
    )


def test_blackbody_band_radiance_matches_the_reference_at_unit_emissivity() -> None:
    expected = band_radiance(
        object_temperature_k=300.0,
        reflected_temperature_k=300.0,
        atmospheric_temperature_k=300.0,
        emissivity=1.0,
        transmittance=1.0,
    )
    assert blackbody_band_radiance(temperature_k=300.0) == pytest.approx(expected, rel=1e-12)


def test_emission_radiance_is_emissivity_times_blackbody_radiance() -> None:
    parameters = derive_surface_parameters(surface=_surface(0.75, 310.0))
    expected = 0.75 * blackbody_band_radiance(temperature_k=310.0)
    assert parameters.emission_radiance_w_sr_m2 == pytest.approx(expected, rel=1e-12)


def test_diffuse_albedo_is_the_kirchhoff_complement_of_emissivity() -> None:
    parameters = derive_surface_parameters(surface=_surface(0.85, 300.0))
    assert parameters.diffuse_albedo == pytest.approx(0.15, rel=1e-12)


def test_emission_plus_reflected_reproduces_the_full_reference_bracket() -> None:
    """The renderer computes eps*B(To) + (1-eps)*B(Tr); that must equal the
    reference at transmittance 1. This is the arithmetic half of SPEC-007's
    first acceptance criterion, checkable without Blender."""
    emissivity, object_k, reflected_k = 0.6, 320.0, 280.0
    parameters = derive_surface_parameters(surface=_surface(emissivity, object_k))
    world_radiance = blackbody_band_radiance(temperature_k=reflected_k)

    renderer_result = (
        parameters.emission_radiance_w_sr_m2 + parameters.diffuse_albedo * world_radiance
    )
    reference = band_radiance(
        object_temperature_k=object_k,
        reflected_temperature_k=reflected_k,
        atmospheric_temperature_k=290.0,
        emissivity=emissivity,
        transmittance=1.0,
    )
    assert renderer_result == pytest.approx(reference, rel=1e-12)


def test_a_blackbody_surface_reflects_nothing() -> None:
    parameters = derive_surface_parameters(surface=_surface(1.0, 300.0))
    assert parameters.diffuse_albedo == pytest.approx(0.0, abs=1e-15)


def test_hotter_surfaces_emit_more_at_equal_emissivity() -> None:
    cold = derive_surface_parameters(surface=_surface(0.95, 280.0))
    hot = derive_surface_parameters(surface=_surface(0.95, 330.0))
    assert hot.emission_radiance_w_sr_m2 > cold.emission_radiance_w_sr_m2


def test_non_physical_temperature_is_rejected() -> None:
    with pytest.raises(ValueError, match="temperature_k"):
        blackbody_band_radiance(temperature_k=-1.0)


def test_parameters_are_frozen() -> None:
    parameters = derive_surface_parameters(surface=_surface(0.9, 300.0))
    with pytest.raises(ValidationError):
        parameters.diffuse_albedo = 0.5  # type: ignore[misc]


def test_a_narrower_band_changes_the_result() -> None:
    wide = derive_surface_parameters(surface=_surface(0.95, 300.0))
    narrow = derive_surface_parameters(
        surface=_surface(0.95, 300.0), wavelength_band_m=(8e-6, 10e-6)
    )
    assert narrow.emission_radiance_w_sr_m2 < wide.emission_radiance_w_sr_m2
    assert isinstance(narrow, ThermalSurfaceParameters)
