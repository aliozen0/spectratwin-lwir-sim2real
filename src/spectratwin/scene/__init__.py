"""Deterministic, renderer-independent procedural scene sampling (SPEC-003).

Produces an inspectable/testable ``SceneDescription`` from a sample seed,
scene config and asset registry. Never launches Blender or writes a
dataset; realization into an actual render is a separate adapter layer.
"""
