"""Training-facing FLIR dataset adapter.

Training never reads FLIR files directly. This module re-scans the same
source root a frozen :class:`~spectratwin.real_data.manifest.DatasetManifest`
was built from, filters the fresh scan down to the manifest's
``sample_ids`` and verifies manifest membership, normalized annotations and
referenced image bytes through the manifest's portable content fingerprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from spectratwin.real_data.adapter import resolve_source_path, scan_flir_dataset
from spectratwin.real_data.manifest import DatasetManifest, build_manifest
from spectratwin.real_data.records import FlirSampleRecord
from spectratwin.real_data.taxonomy import category_id_for

#: FLIR ADAS v2 exports use ``coco.json``, not the generic
#: ``thermal_annotations.json`` default (docs/REAL_DATA_CARD.md).
FLIR_ANNOTATION_FILENAME = "coco.json"


def load_training_records(
    manifest: DatasetManifest, flir_source_root: Path
) -> list[FlirSampleRecord]:
    """Re-scan ``flir_source_root`` and return only the records ``manifest`` names.

    Raises ``ValueError`` if the manifest references a ``sample_id`` a fresh
    scan does not contain, or if the stored fingerprint does not match the
    manifest's identity or content fields, including normalized annotations and
    referenced image SHA-256 values.
    """
    scan_result = scan_flir_dataset(flir_source_root, annotation_filename=FLIR_ANNOTATION_FILENAME)
    scanned_by_id = {r.source_id: r for r in scan_result.records}

    manifest_ids = set(manifest.sample_ids)
    filtered = [scanned_by_id[sid] for sid in manifest.sample_ids if sid in scanned_by_id]
    missing = manifest_ids - scanned_by_id.keys()
    if missing:
        example = sorted(missing)[:5]
        raise ValueError(
            f"manifest role={manifest.role!r} references {len(missing)} sample_id(s) "
            f"missing from a fresh scan of {flir_source_root}: {example}"
        )

    recomputed = build_manifest(manifest.role, filtered, manifest.split_seed)
    if recomputed.fingerprint != manifest.fingerprint:
        raise ValueError(
            f"manifest fingerprint mismatch for role={manifest.role!r}: "
            "the fresh source membership or content does not match the manifest "
            f"(expected {manifest.fingerprint}, got {recomputed.fingerprint})"
        )
    return filtered


class FlirDetectionDataset(Dataset[tuple[Image.Image, dict[str, Any]]]):
    """``(image, coco_style_target)`` pairs an HF image processor accepts.

    Each target has the shape ``{"image_id": int, "annotations": [...]}``
    where each annotation is ``{"image_id", "category_id", "bbox", "area",
    "iscrowd"}`` - the per-image COCO-detection shape ``DetrImageProcessor``
    / ``RTDetrImageProcessor`` consume directly.
    """

    def __init__(self, records: list[FlirSampleRecord], image_root: Path) -> None:
        self._records = records
        self._image_root = image_root

    @property
    def records(self) -> list[FlirSampleRecord]:
        """Records in the exact order ``__getitem__`` indexes them."""
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[Image.Image, dict[str, Any]]:
        record = self._records[index]
        image_path = resolve_source_path(self._image_root, record.relative_image_path)
        image = Image.open(image_path).convert("RGB")

        annotations = []
        for i, ann in enumerate(record.annotations):
            x, y, w, h = ann.bbox_xywh
            annotations.append(
                {
                    "image_id": index,
                    "category_id": category_id_for(ann.project_category),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "id": index * 1000 + i,
                }
            )
        target = {"image_id": index, "annotations": annotations}
        return image, target


def load_training_dataset(
    manifest: DatasetManifest, flir_source_root: Path
) -> FlirDetectionDataset:
    """Build the training-facing dataset for one frozen manifest + source root."""
    records = load_training_records(manifest, flir_source_root)
    return FlirDetectionDataset(records, flir_source_root)
