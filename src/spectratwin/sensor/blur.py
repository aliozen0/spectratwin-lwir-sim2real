"""Optical PSF/blur proxy: separable Gaussian smoothing (SPEC-008).

A small configurable Gaussian approximation, not a full MTF model
(docs/SENSOR_MODEL.md "PSF/blur"). Sigma is expressed in pixel-domain
units, per the same doc's guidance.
"""

from __future__ import annotations

import math

import numpy as np


def _gaussian_kernel_1d(sigma_px: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x**2) / (2.0 * sigma_px**2))
    return kernel / kernel.sum()


def apply_gaussian_blur(frame: np.ndarray, sigma_px: float) -> np.ndarray:
    """Separable Gaussian blur with reflect-padded borders.

    ``sigma_px == 0`` is the toggled-off case: returns an exact copy of
    ``frame`` (SPEC-008 acceptance: "disabling all effects preserves
    reference input").
    """
    if sigma_px < 0:
        raise ValueError(f"sigma_px must be >= 0, got {sigma_px}")
    if frame.ndim != 2:
        raise ValueError(f"frame must be 2D, got shape {frame.shape}")
    if sigma_px == 0:
        return frame.copy()

    radius = max(1, math.ceil(3.0 * sigma_px))
    kernel = _gaussian_kernel_1d(sigma_px, radius)

    padded = np.pad(frame, radius, mode="reflect")
    row_blurred = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), axis=1, arr=padded
    )
    return np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="valid"), axis=0, arr=row_blurred
    )
