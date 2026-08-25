"""Deterministic percentile-clip AGC/display mapping (SPEC-008).

The one required deterministic baseline method
(docs/SENSOR_MODEL.md "AGC": "percentile clipping + normalization").
Randomized-AGC ablation variability is a separate, later policy this
module does not implement.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

#: Frames whose percentile spread is at or below this are treated as
#: constant/near-constant, avoiding a divide-by-zero normalization.
NEAR_CONSTANT_EPSILON = 1e-12


class AgcParameters(BaseModel):
    """Applied AGC parameters, persisted for reproducible display mapping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    low_percentile: float = Field(ge=0.0, le=100.0)
    high_percentile: float = Field(ge=0.0, le=100.0)
    output_low: float
    output_high: float
    input_low: float
    input_high: float
    is_near_constant: bool


def apply_agc(
    frame: np.ndarray,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    output_range: tuple[float, float] = (0.0, 255.0),
) -> tuple[np.ndarray, AgcParameters]:
    """Percentile-clip and linearly rescale ``frame`` into ``output_range``.

    A near-constant frame (percentile spread below
    ``NEAR_CONSTANT_EPSILON``) maps to the midpoint of ``output_range``
    everywhere, never dividing by (near-)zero.
    """
    if high_percentile <= low_percentile:
        raise ValueError(
            f"high_percentile ({high_percentile}) must be > low_percentile ({low_percentile})"
        )
    output_low, output_high = output_range
    if output_high <= output_low:
        raise ValueError(f"output_range max ({output_high}) must be > min ({output_low})")

    input_low = float(np.percentile(frame, low_percentile))
    input_high = float(np.percentile(frame, high_percentile))
    is_near_constant = (input_high - input_low) <= NEAR_CONSTANT_EPSILON

    if is_near_constant:
        midpoint = (output_low + output_high) / 2.0
        mapped = np.full(frame.shape, midpoint, dtype=float)
    else:
        clipped = np.clip(frame, input_low, input_high)
        normalized = (clipped - input_low) / (input_high - input_low)
        mapped = normalized * (output_high - output_low) + output_low

    parameters = AgcParameters(
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        output_low=output_low,
        output_high=output_high,
        input_low=input_low,
        input_high=input_high,
        is_near_constant=is_near_constant,
    )
    return mapped, parameters
