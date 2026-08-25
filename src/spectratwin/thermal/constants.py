"""SI physical constants (2019 CODATA exact defined values)."""

from __future__ import annotations

#: Planck constant, J*s (exact, SI 2019 redefinition).
PLANCK_CONSTANT_J_S = 6.62607015e-34

#: Speed of light in vacuum, m/s (exact).
SPEED_OF_LIGHT_M_S = 2.99792458e8

#: Boltzmann constant, J/K (exact, SI 2019 redefinition).
BOLTZMANN_CONSTANT_J_K = 1.380649e-23

#: Default v1 research LWIR band (docs/THERMAL_MODEL.md "Target band").
#: Not a claim of calibration to any specific sensor's passband.
DEFAULT_LWIR_BAND_M = (8e-6, 14e-6)
