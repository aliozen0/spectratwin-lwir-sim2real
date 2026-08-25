"""Toggleable atmospheric stage applied after rendering (SPEC-007).

``docs/THERMAL_MODEL.md`` ("Atmosphere") requires atmosphere effects to be
toggleable so SPEC-012 can measure their downstream value by ablation. A term
baked into material nodes cannot be ablated, so ``tau`` is applied here, over
the linear rendered image, rather than in the shader::

    L_image = tau * L_render + (1 - tau) * B_band(Ta)

The v1 reference model has no path-length dependence (SPEC-006 non-goals: no
"MODTRAN-class atmospheric simulation"), so this is a global affine map, not a
range-dependent one.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from spectratwin.render.parameters import blackbody_band_radiance
from spectratwin.thermal.constants import DEFAULT_LWIR_BAND_M

ATMOSPHERE_PARAMETERS_SCHEMA_VERSION = "spectratwin-atmosphere-parameters-v1"


class AtmosphereParameters(BaseModel):
    """Applied atmospheric parameters, persisted with the frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = ATMOSPHERE_PARAMETERS_SCHEMA_VERSION
    transmittance: float = Field(ge=0.0, le=1.0)
    atmospheric_temperature_k: float = Field(gt=0.0)
    path_radiance_w_sr_m2: float = Field(ge=0.0)
    #: False when ``transmittance`` is exactly 1.0, i.e. the stage was a no-op.
    #: Recorded so an ablation run states plainly that atmosphere was off
    #: rather than leaving it to be inferred from the value of tau.
    enabled: bool


def apply_atmosphere(
    frame: np.ndarray,
    *,
    transmittance: float,
    atmospheric_temperature_k: float,
    wavelength_band_m: tuple[float, float] = DEFAULT_LWIR_BAND_M,
) -> tuple[np.ndarray, AtmosphereParameters]:
    """Apply the atmospheric term to a linear radiance frame.

    Returns a new array; ``frame`` is not modified. At ``transmittance == 1.0``
    the returned array is bit-identical to the input, so a disabled atmosphere
    cannot perturb ``thermal_raw``.
    """
    if not (0.0 <= transmittance <= 1.0):
        raise ValueError(f"transmittance must be in [0, 1], got {transmittance}")
    if atmospheric_temperature_k <= 0:
        raise ValueError(
            "atmospheric_temperature_k must be positive (absolute), "
            f"got {atmospheric_temperature_k}"
        )

    path_radiance = blackbody_band_radiance(
        temperature_k=atmospheric_temperature_k, wavelength_band_m=wavelength_band_m
    )
    parameters = AtmosphereParameters(
        transmittance=transmittance,
        atmospheric_temperature_k=atmospheric_temperature_k,
        path_radiance_w_sr_m2=path_radiance,
        enabled=transmittance != 1.0,
    )

    if not parameters.enabled:
        return frame.copy(), parameters

    return transmittance * frame + (1.0 - transmittance) * path_radiance, parameters
