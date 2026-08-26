"""Render adapter layer (SPEC-007).

Modules in this package that import ``bpy`` are executed only under
``blenderproc run``. This ``__init__`` and the Blender-free modules must stay
importable in the ordinary project environment, so nothing Blender-bound is
re-exported here (``docs/ARCHITECTURE.md``, "Dependency direction").
"""

from __future__ import annotations

from spectratwin.render.atmosphere import AtmosphereParameters, apply_atmosphere
from spectratwin.render.parameters import (
    ThermalSurfaceParameters,
    blackbody_band_radiance,
    derive_surface_parameters,
)

__all__ = [
    "AtmosphereParameters",
    "ThermalSurfaceParameters",
    "apply_atmosphere",
    "blackbody_band_radiance",
    "derive_surface_parameters",
]
