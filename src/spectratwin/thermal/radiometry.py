"""Planck spectral radiance and band-integrated LWIR radiometry (SPEC-006).

::

    Lλ = tau * [eps * Bλ(To) + (1 - eps) * Bλ(Tr)] + (1 - tau) * Bλ(Ta)

(docs/THERMAL_MODEL.md "Reference image-formation model"). Band radiance is
the numerical integral of this over a wavelength range, optionally weighted
by a configurable sensor response.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from spectratwin.thermal.constants import (
    BOLTZMANN_CONSTANT_J_K,
    DEFAULT_LWIR_BAND_M,
    PLANCK_CONSTANT_J_S,
    SPEED_OF_LIGHT_M_S,
)

#: Maps a wavelength array (meters) to a unitless response array, same shape.
SensorResponse = Callable[[np.ndarray], np.ndarray]

#: NumPy 2.0 renamed ``trapz`` to ``trapezoid`` and removed the old name; the
#: implementation is the same, so this alias changes no numerical result. It
#: exists because ADR-003 makes the renderer and trainer separate runtimes and
#: they do not agree on NumPy: Blender 4.2.1 embeds 1.24.3 (``trapz`` only)
#: while the project environment resolves 2.x (``trapezoid`` only). Core domain
#: code has to import cleanly in both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]


def _uniform_sensor_response(wavelength_m: np.ndarray) -> np.ndarray:
    """Documented default: flat unit response across the configured band."""
    return np.ones_like(wavelength_m)


def planck_spectral_radiance(wavelength_m: float | np.ndarray, temperature_k: float) -> np.ndarray:
    """Planck's law: spectral radiance, W*sr^-1*m^-3 (per meter of wavelength).

    ``wavelength_m`` may be a scalar or array; ``temperature_k`` is a single
    absolute temperature in Kelvin, which MUST be positive.
    """
    wavelength_m = np.asarray(wavelength_m, dtype=float)
    if np.any(wavelength_m <= 0):
        raise ValueError("wavelength_m must be positive")
    if temperature_k <= 0:
        raise ValueError(f"temperature_k must be positive (absolute), got {temperature_k}")

    numerator = 2.0 * PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_S**2
    exponent = (PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_S) / (
        wavelength_m * BOLTZMANN_CONSTANT_J_K * temperature_k
    )
    denominator = wavelength_m**5 * (np.exp(exponent) - 1.0)
    return numerator / denominator


def _validate_unit_interval(value: float, name: str) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def band_radiance(
    *,
    object_temperature_k: float,
    reflected_temperature_k: float,
    atmospheric_temperature_k: float,
    emissivity: float,
    transmittance: float,
    wavelength_band_m: tuple[float, float] = DEFAULT_LWIR_BAND_M,
    sensor_response: SensorResponse = _uniform_sensor_response,
    n_samples: int = 201,
) -> float:
    """Band-integrated radiance, W*sr^-1*m^-2, over ``wavelength_band_m``.

    Numerically integrated with the trapezoidal rule over a fixed
    ``n_samples``-point wavelength grid, so a given input always integrates
    to the same value (deterministic response integration). ``sensor_response``
    defaults to a flat/uniform response — a documented simplification kept
    until source-backed calibration data is available (SPEC-006 Requirements:
    "Generic sensor response assumption MUST be configurable/documented").
    """
    _validate_unit_interval(emissivity, "emissivity")
    _validate_unit_interval(transmittance, "transmittance")
    for name, value in (
        ("object_temperature_k", object_temperature_k),
        ("reflected_temperature_k", reflected_temperature_k),
        ("atmospheric_temperature_k", atmospheric_temperature_k),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive (absolute), got {value}")

    band_min_m, band_max_m = wavelength_band_m
    if band_min_m <= 0 or band_max_m <= 0:
        raise ValueError("wavelength_band_m bounds must be positive")
    if band_max_m <= band_min_m:
        raise ValueError(f"wavelength_band_m max ({band_max_m}) must be > min ({band_min_m})")
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2 for numerical integration")

    wavelengths_m = np.linspace(band_min_m, band_max_m, n_samples)
    spectral_radiance = transmittance * (
        emissivity * planck_spectral_radiance(wavelengths_m, object_temperature_k)
        + (1.0 - emissivity) * planck_spectral_radiance(wavelengths_m, reflected_temperature_k)
    ) + (1.0 - transmittance) * planck_spectral_radiance(wavelengths_m, atmospheric_temperature_k)

    weighted = spectral_radiance * sensor_response(wavelengths_m)
    return float(_trapezoid(weighted, wavelengths_m))
