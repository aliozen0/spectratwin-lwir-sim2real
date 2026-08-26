"""Physically motivated LWIR reference radiometry (SPEC-006).

Pure Python/NumPy, no Blender dependency: the reference equations are the
scientific source of truth (docs/ARCHITECTURE.md "Thermal"); the eventual
Blender/OSL shader is validated against this module, not the other way
around. Does not claim calibration to a named sensor (docs/THERMAL_MODEL.md
"Positioning").
"""
