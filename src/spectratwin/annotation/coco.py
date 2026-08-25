"""Deterministic COCO records built from visible instance masks (SPEC-005)."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from spectratwin.annotation.masks import derive_semantic_map, extract_visible_instances
from spectratwin.annotation.policy import AnnotationPolicy, ExclusionReason
from spectratwin.annotation.rle import UncompressedRle, encode_rle
from spectratwin.real_data.taxonomy import (
    CATEGORY_MAPPING_VERSION,
    PROJECT_CATEGORIES,
    category_id_for,
)

COCO_DOCUMENT_SCHEMA_VERSION = "spectratwin-coco-v1"
ANNOTATION_ID_STRIDE = 10_000


class AnnotationIdError(ValueError):
    """Raised when sample/instance indices cannot form a valid stable ID."""


class CocoWriteError(RuntimeError):
    """Base class for failures that leave the destination document unchanged."""


class CategorySchemaMismatchError(CocoWriteError):
    """Raised when an existing COCO category block differs from the project."""


class CocoIdConflictError(CocoWriteError):
    """Raised when a deterministic ID is reused for different content."""


class CocoValidationError(CocoWriteError):
    """Raised when an incoming or existing document fails internal validation."""


class CocoCategoryRecord(BaseModel):
    """One category from the sole project taxonomy registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int = Field(ge=0, strict=True)
    name: str = Field(min_length=1)
    supercategory: Literal["object"] = "object"


class CocoObjectRecord(BaseModel):
    """Visibility metadata for an expected object, included or excluded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instance_index: int = Field(ge=0, strict=True)
    instance_id: int = Field(ge=1, strict=True)
    category_id: int = Field(ge=0, strict=True)
    category_name: str = Field(min_length=1)
    visible_area_px: int = Field(ge=0, strict=True)
    bbox: tuple[int, int, int, int] | None
    truncated: bool
    included: bool
    exclusion_reason: ExclusionReason | None


class CocoImageRecord(BaseModel):
    """COCO image linkage plus resolved policy and expected-object metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int = Field(ge=0, strict=True)
    file_name: str = Field(min_length=1)
    width: int = Field(gt=0, strict=True)
    height: int = Field(gt=0, strict=True)
    annotation_policy: AnnotationPolicy
    objects: tuple[CocoObjectRecord, ...]

    @field_validator("file_name")
    @classmethod
    def _file_name_is_portable(cls, value: str) -> str:
        path = PurePosixPath(value)
        if "\\" in value or path.is_absolute() or ".." in path.parts or value in {"", "."}:
            raise ValueError("file_name must be a portable relative path")
        return value


class CocoAnnotationRecord(BaseModel):
    """One visible, policy-included instance annotation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int = Field(ge=0, strict=True)
    image_id: int = Field(ge=0, strict=True)
    category_id: int = Field(ge=0, strict=True)
    bbox: tuple[int, int, int, int]
    area: int = Field(gt=0, strict=True)
    segmentation: UncompressedRle
    iscrowd: Literal[0] = 0
    instance_index: int = Field(ge=0, strict=True)
    truncated: bool


class CocoDocument(BaseModel):
    """Strict SpectraTwin COCO document persisted by the guarded writer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["spectratwin-coco-v1"] = COCO_DOCUMENT_SCHEMA_VERSION
    category_mapping_version: str = CATEGORY_MAPPING_VERSION
    images: tuple[CocoImageRecord, ...]
    categories: tuple[CocoCategoryRecord, ...]
    annotations: tuple[CocoAnnotationRecord, ...]

    @model_validator(mode="after")
    def _category_schema_is_current(self) -> Self:
        if self.category_mapping_version != CATEGORY_MAPPING_VERSION:
            raise ValueError("category_mapping_version does not match the project taxonomy")
        if self.categories != project_category_records():
            raise ValueError("categories do not match the ordered project taxonomy")
        return self


@dataclass(frozen=True, slots=True)
class FrameAnnotations:
    """One image record, its annotations and the derived semantic array."""

    image: CocoImageRecord
    annotations: tuple[CocoAnnotationRecord, ...]
    semantic_map: np.ndarray


def project_category_records() -> tuple[CocoCategoryRecord, ...]:
    """Build the exact ordered COCO category block from project taxonomy."""
    return tuple(
        CocoCategoryRecord(id=category_id, name=name)
        for category_id, name in enumerate(PROJECT_CATEGORIES)
    )


def deterministic_annotation_id(*, image_id: int, instance_index: int) -> int:
    """Return the shard-safe annotation ID or raise on stride overflow."""
    if type(image_id) is not int or image_id < 0:
        raise AnnotationIdError("image_id must be a non-negative integer")
    if type(instance_index) is not int or not 0 <= instance_index < ANNOTATION_ID_STRIDE:
        raise AnnotationIdError(
            f"instance_index must be in [0, {ANNOTATION_ID_STRIDE}), got {instance_index!r}"
        )
    return image_id * ANNOTATION_ID_STRIDE + instance_index


def build_frame_annotations(
    *,
    sample_index: int,
    file_name: str,
    instance_map: np.ndarray,
    instance_categories: Mapping[int, str],
    policy: AnnotationPolicy,
) -> FrameAnnotations:
    """Build all SPEC-005 records for one rendered instance map."""
    if type(sample_index) is not int or sample_index < 0:
        raise AnnotationIdError("sample_index must be a non-negative integer")

    semantic_map = derive_semantic_map(instance_map, instance_categories)
    visible_by_index = {
        visible.instance_index: visible for visible in extract_visible_instances(instance_map)
    }
    objects: list[CocoObjectRecord] = []
    annotations: list[CocoAnnotationRecord] = []

    for instance_index, category_name in sorted(instance_categories.items()):
        annotation_id = deterministic_annotation_id(
            image_id=sample_index, instance_index=instance_index
        )
        category_id = category_id_for(category_name)
        visible = visible_by_index.get(instance_index)
        visible_area_px = 0 if visible is None else visible.visible_area_px
        bbox = None if visible is None else visible.bbox_xywh
        truncated = False if visible is None else visible.truncated
        exclusion_reason = policy.exclusion_reason(
            visible_area_px=visible_area_px,
            bbox_xywh=bbox,
        )
        included = exclusion_reason is None
        objects.append(
            CocoObjectRecord(
                instance_index=instance_index,
                instance_id=instance_index + 1,
                category_id=category_id,
                category_name=category_name,
                visible_area_px=visible_area_px,
                bbox=bbox,
                truncated=truncated,
                included=included,
                exclusion_reason=exclusion_reason,
            )
        )
        if included:
            assert bbox is not None
            annotations.append(
                CocoAnnotationRecord(
                    id=annotation_id,
                    image_id=sample_index,
                    category_id=category_id,
                    bbox=bbox,
                    area=visible_area_px,
                    segmentation=encode_rle(instance_map == instance_index + 1),
                    instance_index=instance_index,
                    truncated=truncated,
                )
            )

    height, width = instance_map.shape
    image = CocoImageRecord(
        id=sample_index,
        file_name=file_name,
        width=int(width),
        height=int(height),
        annotation_policy=policy,
        objects=tuple(objects),
    )
    return FrameAnnotations(
        image=image,
        annotations=tuple(annotations),
        semantic_map=semantic_map,
    )


def build_coco_document(frames: Iterable[FrameAnnotations]) -> CocoDocument:
    """Assemble frames into deterministic ID order without rebasing."""
    materialized = tuple(frames)
    images = tuple(sorted((frame.image for frame in materialized), key=lambda image: image.id))
    annotations = tuple(
        sorted(
            (annotation for frame in materialized for annotation in frame.annotations),
            key=lambda annotation: annotation.id,
        )
    )
    return CocoDocument(
        images=images,
        categories=project_category_records(),
        annotations=annotations,
    )


def _expected_category_payload() -> list[dict[str, object]]:
    return [category.model_dump(mode="json") for category in project_category_records()]


def _guard_category_schema(payload: Mapping[str, object]) -> None:
    if (
        payload.get("category_mapping_version") != CATEGORY_MAPPING_VERSION
        or payload.get("categories") != _expected_category_payload()
    ):
        raise CategorySchemaMismatchError(
            "existing ordered category block or mapping version is incompatible"
        )


def _merge_records(
    existing: object,
    incoming: object,
    *,
    record_name: str,
) -> list[dict[str, object]]:
    if not isinstance(existing, list) or not isinstance(incoming, list):
        raise CocoValidationError(f"{record_name} arrays must be JSON lists")
    merged: dict[int, dict[str, object]] = {}
    for raw_record in [*existing, *incoming]:
        if not isinstance(raw_record, dict) or type(raw_record.get("id")) is not int:
            raise CocoValidationError(f"invalid {record_name} record")
        record_id = int(raw_record["id"])
        prior = merged.get(record_id)
        if prior is not None and prior != raw_record:
            raise CocoIdConflictError(f"{record_name[:-1]} id {record_id} has conflicting content")
        merged[record_id] = raw_record
    return [merged[record_id] for record_id in sorted(merged)]


def write_coco_document(path: Path, document: CocoDocument) -> CocoDocument:
    """Validate, strictly merge and atomically persist one COCO document."""
    from spectratwin.annotation.validator import validate_coco_document

    incoming = document.model_dump(mode="json")
    incoming_issues = validate_coco_document(incoming)
    if incoming_issues:
        raise CocoValidationError(f"incoming document has {len(incoming_issues)} violation(s)")

    merged: dict[str, object] = incoming
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CocoValidationError("existing COCO document is unreadable") from exc
        if not isinstance(existing, dict):
            raise CocoValidationError("existing COCO document must be a JSON object")
        _guard_category_schema(existing)
        existing_issues = validate_coco_document(existing)
        if existing_issues:
            raise CocoValidationError(f"existing document has {len(existing_issues)} violation(s)")
        merged = {
            "schema_version": COCO_DOCUMENT_SCHEMA_VERSION,
            "category_mapping_version": CATEGORY_MAPPING_VERSION,
            "images": _merge_records(
                existing.get("images"), incoming["images"], record_name="images"
            ),
            "categories": _expected_category_payload(),
            "annotations": _merge_records(
                existing.get("annotations"),
                incoming["annotations"],
                record_name="annotations",
            ),
        }

    merged_issues = validate_coco_document(merged)
    if merged_issues:
        raise CocoValidationError(f"merged document has {len(merged_issues)} violation(s)")
    # JSON arrays deserialize as lists. Field-level strict integers still
    # reject coercive scalar changes while pydantic freezes list containers as
    # the tuple-based persisted models.
    validated = CocoDocument.model_validate(merged)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(
                validated.model_dump(mode="json"),
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return validated
