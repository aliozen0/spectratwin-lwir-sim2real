"""Thermal sensor processing pipeline stages (SPEC-008).

Pure NumPy, no Blender dependency: every stage is a plain function over a
2D array so it is "testable outside Blender whenever possible" (SPEC-008
Requirements). A stage is "toggled off" by not calling it, or by passing
its neutral parameter (``sigma_px=0`` for blur, ``sigma_read=0`` and
``signal_dependent_gain=0`` for noise) - see each module's tests for the
"disabling preserves reference input" acceptance criterion.

Geometric distortion is intentionally not implemented here yet: SPEC-004
already stores distortion coefficients but notes "sensor distortion may be
applied later," and it is not named in any of SPEC-008's acceptance
criteria (unlike blur/noise/AGC/quantization, which all are). Recording
applied per-stage parameters into a ``FrameRecord`` is a SPEC-009 dataset-
orchestrator integration point that does not exist yet; these functions
return plain arrays so that integration can wrap them later.
"""
