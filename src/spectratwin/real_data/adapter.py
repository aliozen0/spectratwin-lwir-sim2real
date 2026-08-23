"""Scan a local FLIR ADAS checkout into normalized, read-only records.

Never reads/writes anything outside the given source root, never copies
FLIR images/labels elsewhere. The source root is caller-supplied
configuration, never hardcoded here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from spectratwin.real_data.records import (
    FlirAnnotationRecord,
    FlirSampleRecord,
    ScanErrorCategory,
    ScanIssue,
    ScanResult,
)
from spectratwin.real_data.taxonomy import map_category

_TRAILING_DIGITS = re.compile(r"(\d+)$")


def resolve_source_path(root: Path, relative_path: str) -> Path:
    """Resolve one portable source path while confining it to ``root``.

    The caller may supply ``root`` itself through a symlink, but paths stored
    by the dataset must be non-empty relative paths without parent traversal.
    Existing symlinks below the root are accepted only when their resolved
    target remains below the resolved root.
    """
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("source path must be a non-empty string")

    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("source path must be relative and cannot contain '..'")

    try:
        resolved_root = root.resolve(strict=False)
        resolved_path = (resolved_root / candidate).resolve(strict=False)
        resolved_path.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("source path must resolve within the dataset root") from exc
    return resolved_path


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sequence_key(file_name: str, extra_info: dict[str, Any] | None) -> str:
    """Clip/sequence grouping for leakage-safe splitting.

    Prefers the source-backed ``extra_info.video_id`` FLIR provides on
    every image. Falls back to a filename heuristic (strip a trailing
    frame-index run) only when that field is absent - an assumption, not
    a source-backed fact, used solely to keep whole clips together across
    splits.
    """
    if extra_info is not None:
        video_id = extra_info.get("video_id")
        if isinstance(video_id, str) and video_id:
            return video_id

    stem = Path(file_name).stem
    match = _TRAILING_DIGITS.search(stem)
    if match is None or match.start() == 0:
        return stem
    prefix = stem[: match.start()].rstrip("_- ")
    return prefix or stem


def _valid_bbox(bbox: Any, width: int, height: int) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = (float(v) for v in bbox)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (x, y, w, h)):
        return None
    if w <= 0 or h <= 0:
        return None
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return None
    return (x, y, w, h)


def scan_flir_dataset(
    root: Path, annotation_filename: str = "thermal_annotations.json"
) -> ScanResult:
    """Scan ``root`` for a FLIR-style COCO annotation file and its images.

    Missing/corrupt inputs are reported as :class:`ScanIssue` entries with
    stable categories rather than raising, so a partially usable dataset
    still yields whatever normalizes cleanly.
    """
    try:
        annotation_path = resolve_source_path(root, annotation_filename)
    except ValueError:
        issue = ScanIssue(
            source_id=str(annotation_filename),
            category=ScanErrorCategory.INVALID_SOURCE_PATH,
            detail="annotation path must remain within the dataset root",
        )
        return ScanResult(records=(), issues=(issue,))

    if not annotation_path.is_file():
        issue = ScanIssue(
            source_id=annotation_filename,
            category=ScanErrorCategory.MISSING_ANNOTATION_FILE,
            detail=f"annotation file not found below dataset root: {annotation_filename}",
        )
        return ScanResult(records=(), issues=(issue,))

    try:
        payload = json.loads(annotation_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        issue = ScanIssue(
            source_id=str(annotation_path),
            category=ScanErrorCategory.CORRUPT_ANNOTATION_FILE,
            detail=str(exc),
        )
        return ScanResult(records=(), issues=(issue,))

    category_names = {c["id"]: c["name"] for c in payload.get("categories", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in payload.get("annotations", []):
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    records: list[FlirSampleRecord] = []
    issues: list[ScanIssue] = []

    for image in payload.get("images", []):
        file_name = image["file_name"]
        width, height = image["width"], image["height"]
        try:
            image_path = resolve_source_path(root, file_name)
        except ValueError:
            issues.append(
                ScanIssue(
                    source_id=str(file_name),
                    category=ScanErrorCategory.INVALID_SOURCE_PATH,
                    detail="image path must remain within the dataset root",
                )
            )
            continue
        if not image_path.is_file():
            issues.append(
                ScanIssue(
                    source_id=file_name,
                    category=ScanErrorCategory.MISSING_IMAGE,
                    detail=f"image not found below dataset root: {file_name}",
                )
            )
            continue

        record_annotations: list[FlirAnnotationRecord] = []
        sample_has_invalid_bbox = False
        for ann in annotations_by_image.get(image["id"], []):
            category_name = category_names.get(ann["category_id"])
            if category_name is None:
                continue
            project_category = map_category(category_name)
            if project_category is None:
                continue  # unsupported FLIR category: explicitly ignored
            bbox = _valid_bbox(ann.get("bbox"), width, height)
            if bbox is None:
                sample_has_invalid_bbox = True
                issues.append(
                    ScanIssue(
                        source_id=file_name,
                        category=ScanErrorCategory.INVALID_BBOX,
                        detail=f"annotation {ann.get('id')} has invalid bbox {ann.get('bbox')!r}",
                    )
                )
                continue
            record_annotations.append(
                FlirAnnotationRecord(project_category=project_category, bbox_xywh=bbox)
            )

        if sample_has_invalid_bbox:
            continue

        try:
            image_sha256 = _sha256_file(image_path)
        except OSError as exc:
            issues.append(
                ScanIssue(
                    source_id=file_name,
                    category=ScanErrorCategory.UNREADABLE_IMAGE,
                    detail=f"image could not be read: {exc}",
                )
            )
            continue

        records.append(
            FlirSampleRecord(
                source_id=file_name,
                sequence_key=_sequence_key(file_name, image.get("extra_info")),
                relative_image_path=file_name,
                width=width,
                height=height,
                image_sha256=image_sha256,
                annotations=tuple(record_annotations),
            )
        )

    return ScanResult(records=tuple(records), issues=tuple(issues))
