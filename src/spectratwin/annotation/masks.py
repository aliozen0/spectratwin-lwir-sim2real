"""Visible-mask geometry and semantic derivation (SPEC-005)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from spectratwin.real_data.taxonomy import PROJECT_CATEGORIES, category_id_for

SEMANTIC_BACKGROUND_ID = -1


class AnnotationMaskError(ValueError):
    """Raised when an instance map cannot represent the SPEC-005 contract."""


class UnmappedInstanceError(AnnotationMaskError):
    """Raised when a visible instance has no instance-to-category mapping."""


class UnknownCategoryError(AnnotationMaskError):
    """Raised when an instance mapping names a category outside the taxonomy."""


@dataclass(frozen=True, slots=True)
class VisibleInstance:
    """Geometry derived from all visible pixels of one rendered instance."""

    instance_index: int
    instance_id: int
    visible_area_px: int
    bbox_xywh: tuple[int, int, int, int]
    truncated: bool


def validate_instance_map(instance_map: np.ndarray) -> np.ndarray:
    """Return ``instance_map`` after strict structural validation.

    Floating-point maps are decoded and checked at the Blender adapter boundary;
    annotation-domain functions accept integer IDs only.
    """
    array = np.asarray(instance_map)
    if array.ndim != 2:
        raise AnnotationMaskError(f"instance map must be 2D, got shape {array.shape}")
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(array.dtype, np.integer):
        raise AnnotationMaskError(f"instance map must have an integer dtype, got {array.dtype}")
    if np.any(array < 0):
        raise AnnotationMaskError("instance map values must be non-negative")
    return array


def extract_visible_instances(instance_map: np.ndarray) -> tuple[VisibleInstance, ...]:
    """Return visible instance geometry ordered by positive rendered ID."""
    array = validate_instance_map(instance_map)
    height, width = array.shape
    visible: list[VisibleInstance] = []

    for raw_instance_id in np.unique(array):
        instance_id = int(raw_instance_id)
        if instance_id == 0:
            continue
        rows, columns = np.nonzero(array == instance_id)
        x_min = int(columns.min())
        x_max = int(columns.max())
        y_min = int(rows.min())
        y_max = int(rows.max())
        visible.append(
            VisibleInstance(
                instance_index=instance_id - 1,
                instance_id=instance_id,
                visible_area_px=int(rows.size),
                bbox_xywh=(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1),
                truncated=(x_min == 0 or y_min == 0 or x_max == width - 1 or y_max == height - 1),
            )
        )

    return tuple(visible)


def derive_semantic_map(
    instance_map: np.ndarray,
    instance_categories: Mapping[int, str],
) -> np.ndarray:
    """Map rendered instance IDs to zero-based project category IDs.

    The input mapping is keyed by zero-based ``instance_index`` while rendered
    IDs are ``instance_index + 1``. Background remains the signed sentinel -1.
    """
    array = validate_instance_map(instance_map)
    category_ids: dict[int, int] = {}
    for instance_index, category in instance_categories.items():
        if instance_index < 0:
            raise AnnotationMaskError("instance indices must be non-negative")
        if category not in PROJECT_CATEGORIES:
            raise UnknownCategoryError(
                f"unknown project category {category!r}; expected one of {PROJECT_CATEGORIES}"
            )
        category_ids[instance_index] = category_id_for(category)

    visible_ids = {int(value) for value in np.unique(array) if int(value) != 0}
    mapped_ids = {instance_index + 1 for instance_index in category_ids}
    missing = sorted(visible_ids - mapped_ids)
    if missing:
        raise UnmappedInstanceError(f"visible instance_id {missing[0]} has no category mapping")

    semantic = np.full(array.shape, SEMANTIC_BACKGROUND_ID, dtype=np.int16)
    for instance_index, category_id in category_ids.items():
        semantic[array == instance_index + 1] = category_id
    return semantic
