"""Explicit float -> fixed-bit-depth quantization (SPEC-008).

Keeps internal calculations floating point and quantizes only at this
explicit boundary; bit depth and the normalization range are always
recorded by the caller alongside the output, never left implicit
(docs/SENSOR_MODEL.md "Quantization").
"""

from __future__ import annotations

import numpy as np


def quantize_to_bit_depth(
    frame: np.ndarray, *, bit_depth: int, value_range: tuple[float, float]
) -> np.ndarray:
    """Clip ``frame`` to ``value_range`` and quantize to ``bit_depth`` unsigned levels.

    Returns an integer array in ``[0, 2**bit_depth - 1]``, using the
    smallest unsigned NumPy dtype that fits.
    """
    if not (1 <= bit_depth <= 16):
        raise ValueError(f"bit_depth must be in [1, 16], got {bit_depth}")
    low, high = value_range
    if high <= low:
        raise ValueError(f"value_range max ({high}) must be > min ({low})")

    max_level = (2**bit_depth) - 1
    clipped = np.clip(frame, low, high)
    normalized = (clipped - low) / (high - low)
    levels = np.round(normalized * max_level)
    dtype = np.uint8 if bit_depth <= 8 else np.uint16
    return levels.astype(dtype)
