"""Blender-free ground-truth geometry and COCO contracts (SPEC-005)."""

from __future__ import annotations

from spectratwin.annotation.masks import (
    VisibleInstance,
    derive_semantic_map,
    extract_visible_instances,
)
from spectratwin.annotation.policy import AnnotationPolicy, ExclusionReason

__all__ = [
    "AnnotationPolicy",
    "ExclusionReason",
    "VisibleInstance",
    "derive_semantic_map",
    "extract_visible_instances",
]
