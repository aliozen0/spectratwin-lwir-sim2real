"""Named LWIR thermal material classes (SPEC-007).

``docs/THERMAL_MODEL.md`` ("Thermal attributes") requires a thermal material
layer separate from visible PBR appearance, where each class defines
"source/assumption notes and distributions rather than one undocumented
constant".

Every value here is a **configuration prior**, not a measurement made for this
project and not calibration data (ADR-002). :class:`EmissivityBasis` makes that
explicit in the type rather than in a comment, so a later source-backed or
measured value cannot quietly inherit the credibility of a citation it does not
have.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

THERMAL_MATERIAL_SCHEMA_VERSION = "spectratwin-thermal-material-v1"


class EmissivityBasis(StrEnum):
    """Where a material's emissivity range came from."""

    #: Conventional LWIR range for the material family. No specific source
    #: document is cited, and none is implied.
    LITERATURE_TYPICAL = "literature-typical"
    #: Traceable to a named, recorded source.
    SOURCE_BACKED = "source-backed"
    #: Measured by this project against a documented procedure.
    MEASURED = "measured"


class ResolvedThermalSurface(BaseModel):
    """One material's drawn emissivity and absolute surface temperature."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    material_name: str = Field(min_length=1)
    emissivity: float = Field(ge=0.0, le=1.0)
    temperature_k: float = Field(gt=0.0)


class ThermalMaterial(BaseModel):
    """A named thermal class with bounded emissivity/temperature priors."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = THERMAL_MATERIAL_SCHEMA_VERSION
    name: str = Field(min_length=1)
    emissivity_min: float = Field(ge=0.0, le=1.0)
    emissivity_max: float = Field(ge=0.0, le=1.0)
    temperature_offset_k_min: float
    temperature_offset_k_max: float
    basis: EmissivityBasis
    notes: str = Field(min_length=20)

    @model_validator(mode="after")
    def _ranges_are_ordered(self) -> ThermalMaterial:
        if self.emissivity_max < self.emissivity_min:
            raise ValueError(
                f"emissivity_max ({self.emissivity_max}) must be >= "
                f"emissivity_min ({self.emissivity_min})"
            )
        if self.temperature_offset_k_max < self.temperature_offset_k_min:
            raise ValueError(
                f"temperature_offset_k_max ({self.temperature_offset_k_max}) must be >= "
                f"temperature_offset_k_min ({self.temperature_offset_k_min})"
            )
        return self

    def draw(
        self, rng: np.random.Generator, *, ambient_temperature_k: float
    ) -> ResolvedThermalSurface:
        """Sample one emissivity and absolute temperature from this class.

        The temperature offset is relative to scene ambient, per
        ``docs/THERMAL_MODEL.md`` ("Temperature model": "Define
        scene-conditioned distributions"), so one material behaves
        consistently across warm and cold scenes.
        """
        if ambient_temperature_k <= 0:
            raise ValueError(
                f"ambient_temperature_k must be positive (absolute), got {ambient_temperature_k}"
            )

        emissivity = float(rng.uniform(self.emissivity_min, self.emissivity_max))
        offset_k = float(rng.uniform(self.temperature_offset_k_min, self.temperature_offset_k_max))
        temperature_k = ambient_temperature_k + offset_k
        if temperature_k <= 0:
            raise ValueError(
                f"drawn temperature_k must be positive (absolute), got {temperature_k}"
            )

        return ResolvedThermalSurface(
            material_name=self.name,
            emissivity=emissivity,
            temperature_k=temperature_k,
        )


def _material(
    name: str,
    *,
    emissivity: tuple[float, float],
    offset_k: tuple[float, float],
    notes: str,
) -> ThermalMaterial:
    return ThermalMaterial(
        name=name,
        emissivity_min=emissivity[0],
        emissivity_max=emissivity[1],
        temperature_offset_k_min=offset_k[0],
        temperature_offset_k_max=offset_k[1],
        basis=EmissivityBasis.LITERATURE_TYPICAL,
        notes=notes,
    )


#: The material families named in docs/THERMAL_MODEL.md "Thermal attributes".
#: Temperature offsets are relative to scene ambient and encode the ordinary
#: daytime-urban expectation that solar-loaded road surfaces run hotter than air
#: and that skin is regulated near body temperature. They are priors to be
#: revised against real-data statistics, not fitted values.
THERMAL_MATERIALS: dict[str, ThermalMaterial] = {
    material.name: material
    for material in (
        _material(
            "asphalt",
            emissivity=(0.92, 0.96),
            offset_k=(0.0, 20.0),
            notes=(
                "Rough mineral/bitumen surface, treated as a high-emissivity dielectric. "
                "Large positive offset covers solar loading on an open roadway."
            ),
        ),
        _material(
            "concrete",
            emissivity=(0.90, 0.95),
            offset_k=(0.0, 15.0),
            notes=(
                "Rough mineral surface similar to asphalt but lighter, so a smaller "
                "assumed solar-loading offset. High-emissivity dielectric."
            ),
        ),
        _material(
            "painted_metal",
            emissivity=(0.85, 0.95),
            offset_k=(-2.0, 25.0),
            notes=(
                "Vehicle bodywork. Paint dominates the LWIR response, so the range is "
                "dielectric-like rather than bare-metal-like; bare polished metal would "
                "be far lower and is deliberately not modelled as one class here."
            ),
        ),
        _material(
            "rubber",
            emissivity=(0.90, 0.96),
            offset_k=(0.0, 30.0),
            notes=(
                "Tyres. High-emissivity elastomer with a wide positive offset because "
                "rolling and braking add heat beyond ambient and solar loading."
            ),
        ),
        _material(
            "automotive_glass",
            emissivity=(0.80, 0.92),
            offset_k=(-3.0, 15.0),
            notes=(
                "Windscreens and windows. Modelled with the lowest emissivity in the "
                "registry, which is what makes reflected sky temperature visible on "
                "glazing; it is the material this project most expects to revise."
            ),
        ),
        _material(
            "fabric",
            emissivity=(0.92, 0.98),
            offset_k=(2.0, 12.0),
            notes=(
                "Clothing proxy. High-emissivity textile sitting between skin and air, "
                "so it is warmer than ambient but cooler than exposed skin."
            ),
        ),
        _material(
            "skin",
            emissivity=(0.96, 0.99),
            offset_k=(8.0, 16.0),
            notes=(
                "Exposed skin proxy. Near-blackbody in the LWIR and thermoregulated, so "
                "the offset assumes an ambient near 293 K rather than tracking it "
                "without limit; revisit for hot-ambient scenes."
            ),
        ),
        _material(
            "vegetation",
            emissivity=(0.94, 0.99),
            offset_k=(-2.0, 6.0),
            notes=(
                "Foliage. Near-blackbody in the LWIR; transpiration keeps it close to "
                "or slightly below air temperature despite solar loading."
            ),
        ),
    )
}


def get_thermal_material(name: str) -> ThermalMaterial:
    """Return one registered material, or fail loudly with the known names."""
    try:
        return THERMAL_MATERIALS[name]
    except KeyError as error:
        known = ", ".join(sorted(THERMAL_MATERIALS))
        raise KeyError(f"unknown thermal material {name!r}; known materials: {known}") from error
