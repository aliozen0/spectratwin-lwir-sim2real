"""Per-frame thermal output contract (SPEC-007).

``docs/THERMAL_MODEL.md`` ("Raw versus display image") requires both a
high-bit-depth linear ``thermal_raw`` proxy and a ``thermal_agc`` display
mapping, and requires that the raw proxy is not destroyed to make a more
presentable image. This module carries the metadata that makes both
reproducible.

Blender-free on purpose: the metadata contract is testable in the ordinary gate,
and only the EXR write itself needs Blender.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from spectratwin.render.atmosphere import AtmosphereParameters
from spectratwin.render.parameters import ThermalSurfaceParameters
from spectratwin.render.settings import RenderSettings
from spectratwin.sensor.agc import AgcParameters

THERMAL_FRAME_SCHEMA_VERSION = "spectratwin-thermal-frame-v1"


class ThermalFrameMetadata(BaseModel):
    """Everything needed to explain or reproduce one rendered thermal frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = THERMAL_FRAME_SCHEMA_VERSION
    sample_seed: int = Field(ge=0)
    master_seed: int = Field(ge=0)
    ambient_temperature_k: float = Field(gt=0.0)
    sky_temperature_k: float = Field(gt=0.0)
    render_settings: RenderSettings
    atmosphere: AtmosphereParameters
    agc: AgcParameters
    surfaces: tuple[ThermalSurfaceParameters, ...]
    #: Renderer identity, e.g. ``blenderproc-2.8.0/blender-4.2.1``. Recorded
    #: because a renderer upgrade can change pixel values, and a surprising
    #: frame should be explainable without re-running it.
    renderer_identity: str = Field(min_length=1)


def build_frame_metadata(
    *,
    sample_seed: int,
    master_seed: int,
    ambient_temperature_k: float,
    sky_temperature_k: float,
    render_settings: RenderSettings,
    atmosphere: AtmosphereParameters,
    agc: AgcParameters,
    surfaces: tuple[ThermalSurfaceParameters, ...],
    renderer_identity: str,
) -> ThermalFrameMetadata:
    """Assemble the per-frame record. Every argument is required."""
    return ThermalFrameMetadata(
        sample_seed=sample_seed,
        master_seed=master_seed,
        ambient_temperature_k=ambient_temperature_k,
        sky_temperature_k=sky_temperature_k,
        render_settings=render_settings,
        atmosphere=atmosphere,
        agc=agc,
        surfaces=surfaces,
        renderer_identity=renderer_identity,
    )
