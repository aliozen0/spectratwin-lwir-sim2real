"""Thermal surface parameters for the Cycles material mapping (SPEC-007).

ADR-010 fixes the physics to one implementation. Every Planck evaluation in the
render path goes through :func:`blackbody_band_radiance`, which calls the
SPEC-006 reference; nothing in the renderer re-derives it.

The reference model is::

    L = tau * [ eps * B(To) + (1 - eps) * B(Tr) ] + (1 - tau) * B(Ta)

This module produces the two surface coefficients of the bracket. Cycles
supplies the incident radiance that multiplies ``diffuse_albedo``, and the
``tau`` terms are applied separately by :mod:`spectratwin.render.atmosphere`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from spectratwin.thermal.constants import DEFAULT_LWIR_BAND_M
from spectratwin.thermal.materials import ResolvedThermalSurface
from spectratwin.thermal.radiometry import band_radiance

THERMAL_SURFACE_PARAMETERS_SCHEMA_VERSION = "spectratwin-thermal-surface-parameters-v1"


def blackbody_band_radiance(
    *,
    temperature_k: float,
    wavelength_band_m: tuple[float, float] = DEFAULT_LWIR_BAND_M,
) -> float:
    """Band-integrated blackbody radiance ``B_band(T)``, W*sr^-1*m^-2.

    Calls the SPEC-006 reference with ``emissivity=1.0`` and
    ``transmittance=1.0``, which collapses the reflected and atmospheric terms
    exactly, so the reflected/atmospheric temperatures passed are arbitrary.
    ``temperature_k`` is passed for all three to avoid inventing unused
    sentinel values.
    """
    return band_radiance(
        object_temperature_k=temperature_k,
        reflected_temperature_k=temperature_k,
        atmospheric_temperature_k=temperature_k,
        emissivity=1.0,
        transmittance=1.0,
        wavelength_band_m=wavelength_band_m,
    )


class ThermalSurfaceParameters(BaseModel):
    """Cycles-facing coefficients for one thermal surface."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = THERMAL_SURFACE_PARAMETERS_SCHEMA_VERSION
    material_name: str = Field(min_length=1)
    emissivity: float = Field(ge=0.0, le=1.0)
    temperature_k: float = Field(gt=0.0)
    #: Emission node strength: ``eps * B_band(To)``, W*sr^-1*m^-2.
    emission_radiance_w_sr_m2: float = Field(ge=0.0)
    #: Diffuse BSDF albedo: ``1 - eps``, by Kirchhoff's law for an opaque
    #: surface. Cycles multiplies this by the incident radiance it solves.
    diffuse_albedo: float = Field(ge=0.0, le=1.0)


def derive_surface_parameters(
    *,
    surface: ResolvedThermalSurface,
    wavelength_band_m: tuple[float, float] = DEFAULT_LWIR_BAND_M,
) -> ThermalSurfaceParameters:
    """Map one resolved surface onto its Emission/Diffuse coefficients."""
    blackbody = blackbody_band_radiance(
        temperature_k=surface.temperature_k, wavelength_band_m=wavelength_band_m
    )
    return ThermalSurfaceParameters(
        material_name=surface.material_name,
        emissivity=surface.emissivity,
        temperature_k=surface.temperature_k,
        emission_radiance_w_sr_m2=surface.emissivity * blackbody,
        diffuse_albedo=1.0 - surface.emissivity,
    )
