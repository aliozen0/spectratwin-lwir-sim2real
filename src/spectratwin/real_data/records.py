"""Normalized, read-only FLIR sample records and scan issue reporting."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ScanErrorCategory(StrEnum):
    MISSING_ANNOTATION_FILE = "missing_annotation_file"
    CORRUPT_ANNOTATION_FILE = "corrupt_annotation_file"
    INVALID_SOURCE_PATH = "invalid_source_path"
    MISSING_IMAGE = "missing_image"
    UNREADABLE_IMAGE = "unreadable_image"
    INVALID_BBOX = "invalid_bbox"


class ScanIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    category: ScanErrorCategory
    detail: str


class FlirAnnotationRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_category: str
    bbox_xywh: tuple[float, float, float, float]


class FlirSampleRecord(BaseModel):
    """One normalized, valid FLIR frame. Paths are relative to the source root."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    sequence_key: str
    relative_image_path: str
    width: int
    height: int
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotations: tuple[FlirAnnotationRecord, ...]


class ScanResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    records: tuple[FlirSampleRecord, ...]
    issues: tuple[ScanIssue, ...]
