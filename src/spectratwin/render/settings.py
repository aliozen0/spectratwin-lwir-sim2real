"""Explicit, persisted Cycles render settings (SPEC-007).

Blender's defaults are tuned for pleasing pictures, not radiometry. Filmic or
AGX view transforms, denoising and light clamping all silently change pixel
values, and each of them would make a thermal render look plausible while being
wrong. SPEC-007 therefore requires every setting that affects a pixel to be set
explicitly; this model is what gets set and what gets persisted alongside the
frame.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RENDER_SETTINGS_SCHEMA_VERSION = "spectratwin-render-settings-v1"

RenderDevice = Literal["CPU", "GPU"]


class RenderSettings(BaseModel):
    """Resolved renderer configuration for one frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = RENDER_SETTINGS_SCHEMA_VERSION
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    samples: int = Field(gt=0)
    max_bounces: int = Field(ge=0)
    diffuse_bounces: int = Field(ge=0)
    seed: int = Field(ge=0)
    device: RenderDevice = "CPU"
    #: Cycles reconstruction-filter width in pixels. Kept near zero for
    #: verification renders so a pixel is not blended with its neighbours;
    #: production frames may widen it.
    filter_width_px: float = Field(gt=0.0, default=0.01)

    @model_validator(mode="after")
    def _diffuse_bounces_fit_inside_max(self) -> RenderSettings:
        if self.diffuse_bounces > self.max_bounces:
            raise ValueError(
                f"diffuse_bounces ({self.diffuse_bounces}) must be <= "
                f"max_bounces ({self.max_bounces})"
            )
        return self

    @classmethod
    def for_reference_check(
        cls, *, width_px: int, height_px: int, seed: int, diffuse_bounces: int
    ) -> RenderSettings:
        """Settings for a numerical comparison against the CPU reference.

        CPU device and a near-zero filter width, so the comparison is not
        confounded by GPU floating-point differences or by the reconstruction
        filter blending neighbouring pixels. ``max_bounces`` follows
        ``diffuse_bounces`` so that ``diffuse_bounces=0`` really does isolate
        the emissive term.
        """
        return cls(
            width_px=width_px,
            height_px=height_px,
            samples=16,
            max_bounces=diffuse_bounces,
            diffuse_bounces=diffuse_bounces,
            seed=seed,
            device="CPU",
            filter_width_px=0.01,
        )
