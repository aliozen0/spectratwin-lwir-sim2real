"""Signal-independent and signal-dependent sensor noise proxies (SPEC-008).

Every random draw derives from a caller-supplied ``numpy.random.Generator``
(SPEC-008 Requirements: "Stochastic effects use explicit sample/subsystem
RNG") - callers derive it from the project master seed via
:func:`spectratwin.randomness.seed.new_generator`, one label path per
sample, so replay with the same seed is exact.
"""

from __future__ import annotations

import numpy as np


def apply_sensor_noise(
    frame: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma_read: float = 0.0,
    signal_dependent_gain: float = 0.0,
) -> np.ndarray:
    """Add ``Normal(0, sigma_read + signal_dependent_gain * |frame|)`` noise per pixel.

    ``sigma_read=0`` and ``signal_dependent_gain=0`` is the toggled-off
    case: returns an exact copy of ``frame``, without drawing from ``rng``.
    """
    if sigma_read < 0:
        raise ValueError(f"sigma_read must be >= 0, got {sigma_read}")
    if signal_dependent_gain < 0:
        raise ValueError(f"signal_dependent_gain must be >= 0, got {signal_dependent_gain}")

    if sigma_read == 0.0 and signal_dependent_gain == 0.0:
        return frame.copy()

    std_px = sigma_read + signal_dependent_gain * np.abs(frame)
    noise = rng.normal(loc=0.0, scale=1.0, size=frame.shape) * std_px
    return frame + noise
