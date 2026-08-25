"""Internal validation over untrusted COCO-shaped documents (SPEC-005)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import TypeGuard

import numpy as np
from pydantic import ValidationError

from spectratwin.annotation.coco import (
    ANNOTATION_ID_STRIDE,
    COCO_DOCUMENT_SCHEMA_VERSION,
    deterministic_annotation_id,
    project_category_records,
)
from spectratwin.annotation.policy import AnnotationPolicy
from spectratwin.annotation.rle import RleError, decode_rle
from spectratwin.real_data.taxonomy import CATEGORY_MAPPING_VERSION, PROJECT_CATEGORIES


class ValidationCode(StrEnum):
    """Stable machine-readable classes for annotation contract violations."""

    DOCUMENT_SHAPE = "document_shape"
    SCHEMA_VERSION = "schema_version"
    CATEGORY_SCHEMA = "category_schema"
    DUPLICATE_IMAGE_ID = "duplicate_image_id"
    DUPLICATE_ANNOTATION_ID = "duplicate_annotation_id"
    IMAGE_GEOMETRY = "image_geometry"
    IMAGE_PATH = "image_path"
    POLICY = "policy"
    OBJECT_METADATA = "object_metadata"
    ANNOTATION_REFERENCE = "annotation_reference"
    CATEGORY_REFERENCE = "category_reference"
    ANNOTATION_ID = "annotation_id"
    BBOX_GEOMETRY = "bbox_geometry"
    AREA = "area"
    SEGMENTATION = "segmentation"
    MASK_GEOMETRY = "mask_geometry"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One validator finding with stable code and document location."""

    code: ValidationCode
    location: str
    detail: str


@dataclass(frozen=True, slots=True)
class _ImageContext:
    width: int
    height: int
    policy: AnnotationPolicy | None
    objects: Mapping[int, Mapping[str, object]]


def _is_int(value: object, *, minimum: int | None = None) -> TypeGuard[int]:
    return type(value) is int and (minimum is None or value >= minimum)


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping)


def _portable_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and value != "."


def _bbox(
    value: object,
    *,
    width: int | None,
    height: int | None,
) -> tuple[int, int, int, int] | None:
    if not _is_sequence(value) or len(value) != 4:
        return None
    x, y, box_width, box_height = value
    if not (_is_int(x) and _is_int(y) and _is_int(box_width) and _is_int(box_height)):
        return None
    if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
        return None
    if width is not None and height is not None:
        if x + box_width > width or y + box_height > height:
            return None
    return (x, y, box_width, box_height)


def validate_coco_document(document: object) -> tuple[ValidationIssue, ...]:
    """Return all safely detectable SPEC-005 violations in ``document``."""
    issues: list[ValidationIssue] = []

    def add(code: ValidationCode, location: str, detail: str) -> None:
        issues.append(ValidationIssue(code=code, location=location, detail=detail))

    if not _is_mapping(document):
        return (
            ValidationIssue(
                code=ValidationCode.DOCUMENT_SHAPE,
                location="$",
                detail="COCO document must be a JSON object",
            ),
        )

    if document.get("schema_version") != COCO_DOCUMENT_SCHEMA_VERSION:
        add(ValidationCode.SCHEMA_VERSION, "schema_version", "unsupported document schema")

    expected_categories = [
        category.model_dump(mode="json") for category in project_category_records()
    ]
    if (
        document.get("category_mapping_version") != CATEGORY_MAPPING_VERSION
        or document.get("categories") != expected_categories
    ):
        add(
            ValidationCode.CATEGORY_SCHEMA,
            "categories",
            "ordered categories or mapping version differ from the project taxonomy",
        )

    raw_images = document.get("images")
    raw_annotations = document.get("annotations")
    if not _is_sequence(raw_images):
        add(ValidationCode.DOCUMENT_SHAPE, "images", "images must be an array")
        images: Sequence[object] = ()
    else:
        images = raw_images
    if not _is_sequence(raw_annotations):
        add(ValidationCode.DOCUMENT_SHAPE, "annotations", "annotations must be an array")
        annotations: Sequence[object] = ()
    else:
        annotations = raw_annotations

    image_contexts: dict[int, _ImageContext] = {}
    seen_image_ids: set[int] = set()
    for image_position, raw_image in enumerate(images):
        location = f"images[{image_position}]"
        if not _is_mapping(raw_image):
            add(ValidationCode.DOCUMENT_SHAPE, location, "image record must be an object")
            continue

        image_id_value = raw_image.get("id")
        image_id = image_id_value if _is_int(image_id_value, minimum=0) else None
        if image_id is None:
            add(ValidationCode.IMAGE_GEOMETRY, f"{location}.id", "invalid image id")
        elif image_id in seen_image_ids:
            add(
                ValidationCode.DUPLICATE_IMAGE_ID,
                f"{location}.id",
                f"duplicate image id {image_id}",
            )
        else:
            seen_image_ids.add(image_id)

        width_value = raw_image.get("width")
        height_value = raw_image.get("height")
        width = width_value if _is_int(width_value, minimum=1) else None
        height = height_value if _is_int(height_value, minimum=1) else None
        if width is None or height is None:
            add(
                ValidationCode.IMAGE_GEOMETRY,
                location,
                "image width and height must be positive integers",
            )
        if not _portable_relative_path(raw_image.get("file_name")):
            add(ValidationCode.IMAGE_PATH, f"{location}.file_name", "invalid portable path")

        raw_policy = raw_image.get("annotation_policy")
        policy: AnnotationPolicy | None = None
        try:
            policy = AnnotationPolicy.model_validate(raw_policy, strict=True)
        except (TypeError, ValidationError):
            add(ValidationCode.POLICY, f"{location}.annotation_policy", "invalid policy")

        raw_objects = raw_image.get("objects")
        object_by_index: dict[int, Mapping[str, object]] = {}
        if not _is_sequence(raw_objects):
            add(ValidationCode.OBJECT_METADATA, f"{location}.objects", "objects must be an array")
            objects: Sequence[object] = ()
        else:
            objects = raw_objects

        for object_position, raw_object in enumerate(objects):
            object_location = f"{location}.objects[{object_position}]"
            if not _is_mapping(raw_object):
                add(ValidationCode.OBJECT_METADATA, object_location, "object must be a record")
                continue
            index_value = raw_object.get("instance_index")
            if not _is_int(index_value, minimum=0) or index_value >= ANNOTATION_ID_STRIDE:
                add(ValidationCode.OBJECT_METADATA, object_location, "invalid instance_index")
                continue
            instance_index = index_value
            if instance_index in object_by_index:
                add(ValidationCode.OBJECT_METADATA, object_location, "duplicate instance_index")
            object_by_index[instance_index] = raw_object

            category_id_value = raw_object.get("category_id")
            category_name = raw_object.get("category_name")
            category_valid = (
                _is_int(category_id_value, minimum=0)
                and category_id_value < len(PROJECT_CATEGORIES)
                and category_name == PROJECT_CATEGORIES[category_id_value]
            )
            visible_area_value = raw_object.get("visible_area_px")
            visible_area = visible_area_value if _is_int(visible_area_value, minimum=0) else None
            object_bbox_value = raw_object.get("bbox")
            object_bbox = (
                None
                if object_bbox_value is None
                else _bbox(object_bbox_value, width=width, height=height)
            )
            instance_id_valid = raw_object.get("instance_id") == instance_index + 1
            flags_valid = (
                type(raw_object.get("truncated")) is bool
                and type(raw_object.get("included")) is bool
            )
            if (
                not category_valid
                or visible_area is None
                or (object_bbox_value is not None and object_bbox is None)
                or not instance_id_valid
                or not flags_valid
            ):
                add(ValidationCode.OBJECT_METADATA, object_location, "invalid object fields")

            if policy is not None and visible_area is not None:
                try:
                    expected_reason = policy.exclusion_reason(
                        visible_area_px=visible_area,
                        bbox_xywh=object_bbox,
                    )
                except ValueError:
                    add(
                        ValidationCode.OBJECT_METADATA,
                        object_location,
                        "visible area and bbox are inconsistent",
                    )
                else:
                    expected_value = None if expected_reason is None else expected_reason.value
                    if (
                        raw_object.get("included") is not (expected_reason is None)
                        or raw_object.get("exclusion_reason") != expected_value
                    ):
                        add(
                            ValidationCode.OBJECT_METADATA,
                            object_location,
                            "inclusion metadata disagrees with annotation policy",
                        )

        if image_id is not None and width is not None and height is not None:
            image_contexts[image_id] = _ImageContext(
                width=width,
                height=height,
                policy=policy,
                objects=object_by_index,
            )

    seen_annotation_ids: set[int] = set()
    annotated_instances: dict[int, set[int]] = {}
    for annotation_position, raw_annotation in enumerate(annotations):
        location = f"annotations[{annotation_position}]"
        if not _is_mapping(raw_annotation):
            add(ValidationCode.DOCUMENT_SHAPE, location, "annotation must be an object")
            continue

        annotation_id_value = raw_annotation.get("id")
        annotation_id = annotation_id_value if _is_int(annotation_id_value, minimum=0) else None
        if annotation_id is None:
            add(ValidationCode.ANNOTATION_ID, f"{location}.id", "invalid annotation id")
        elif annotation_id in seen_annotation_ids:
            add(
                ValidationCode.DUPLICATE_ANNOTATION_ID,
                f"{location}.id",
                f"duplicate annotation id {annotation_id}",
            )
        else:
            seen_annotation_ids.add(annotation_id)

        image_id_value = raw_annotation.get("image_id")
        image_id = image_id_value if _is_int(image_id_value, minimum=0) else None
        image_context = None if image_id is None else image_contexts.get(image_id)
        if image_context is None:
            add(
                ValidationCode.ANNOTATION_REFERENCE,
                f"{location}.image_id",
                "annotation references a missing image",
            )

        category_id_value = raw_annotation.get("category_id")
        category_id = category_id_value if _is_int(category_id_value, minimum=0) else None
        if category_id is None or category_id >= len(PROJECT_CATEGORIES):
            add(
                ValidationCode.CATEGORY_REFERENCE,
                f"{location}.category_id",
                "annotation references a missing category",
            )

        instance_index_value = raw_annotation.get("instance_index")
        instance_index = (
            instance_index_value
            if _is_int(instance_index_value, minimum=0)
            and instance_index_value < ANNOTATION_ID_STRIDE
            else None
        )
        if image_id is not None and instance_index is not None and annotation_id is not None:
            if annotation_id != deterministic_annotation_id(
                image_id=image_id, instance_index=instance_index
            ):
                add(ValidationCode.ANNOTATION_ID, f"{location}.id", "ID formula mismatch")

        bbox = _bbox(
            raw_annotation.get("bbox"),
            width=None if image_context is None else image_context.width,
            height=None if image_context is None else image_context.height,
        )
        if bbox is None:
            add(ValidationCode.BBOX_GEOMETRY, f"{location}.bbox", "invalid bbox")

        area_value = raw_annotation.get("area")
        area = area_value if _is_int(area_value, minimum=1) else None
        if area is None:
            add(ValidationCode.AREA, f"{location}.area", "area must be a positive integer")
        if raw_annotation.get("iscrowd") != 0:
            add(ValidationCode.DOCUMENT_SHAPE, f"{location}.iscrowd", "iscrowd must be zero")

        mask: np.ndarray | None = None
        segmentation = raw_annotation.get("segmentation")
        if not _is_mapping(segmentation):
            add(ValidationCode.SEGMENTATION, f"{location}.segmentation", "RLE must be an object")
        else:
            try:
                mask = decode_rle(segmentation)
            except RleError:
                add(ValidationCode.SEGMENTATION, f"{location}.segmentation", "invalid RLE")

        if mask is not None:
            if image_context is not None and mask.shape != (
                image_context.height,
                image_context.width,
            ):
                add(ValidationCode.SEGMENTATION, f"{location}.segmentation", "size mismatch")
            rows, columns = np.nonzero(mask)
            if rows.size == 0:
                add(ValidationCode.MASK_GEOMETRY, location, "annotation mask is empty")
            else:
                expected_bbox = (
                    int(columns.min()),
                    int(rows.min()),
                    int(columns.max() - columns.min() + 1),
                    int(rows.max() - rows.min() + 1),
                )
                expected_truncated = (
                    columns.min() == 0
                    or rows.min() == 0
                    or columns.max() == mask.shape[1] - 1
                    or rows.max() == mask.shape[0] - 1
                )
                if bbox != expected_bbox or raw_annotation.get("truncated") is not bool(
                    expected_truncated
                ):
                    add(
                        ValidationCode.MASK_GEOMETRY,
                        location,
                        "bbox/truncation is not tight to the visible mask",
                    )
                if area != int(rows.size):
                    add(ValidationCode.AREA, f"{location}.area", "area differs from mask pixels")

        if image_context is not None and image_id is not None and instance_index is not None:
            annotated_instances.setdefault(image_id, set()).add(instance_index)
            object_record = image_context.objects.get(instance_index)
            if object_record is None:
                add(ValidationCode.OBJECT_METADATA, location, "missing object metadata")
            elif (
                object_record.get("included") is not True
                or object_record.get("category_id") != category_id
                or object_record.get("bbox") != raw_annotation.get("bbox")
                or object_record.get("visible_area_px") != area
                or object_record.get("truncated") != raw_annotation.get("truncated")
            ):
                add(
                    ValidationCode.OBJECT_METADATA,
                    location,
                    "annotation disagrees with object metadata",
                )

    for image_id, context in image_contexts.items():
        present = annotated_instances.get(image_id, set())
        for instance_index, object_record in context.objects.items():
            should_exist = object_record.get("included") is True
            if should_exist != (instance_index in present):
                add(
                    ValidationCode.OBJECT_METADATA,
                    f"images[id={image_id}].objects[instance_index={instance_index}]",
                    "included flag disagrees with annotation presence",
                )

    return tuple(issues)
