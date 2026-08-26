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

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as tv_transforms

from spectratwin.randomness.seed import derive_subseed
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


def _record_index_getter(inputs: Any) -> torch.Tensor:
    """``SanitizeBoundingBoxes`` labels_getter: filter our index tensor in
    lockstep with whatever boxes ``RandomIoUCrop``/etc. drop, so dropped
    boxes' original annotation dicts (category/iscrowd/id) can still be
    found afterward."""
    return inputs[1]["record_index"]


#: Mirrors the cited RT-DETR/RT-DETRv2 recipe's ``dataloader.yml`` per-sample
#: ops (``RandomPhotometricDistort``, ``RandomZoomOut``, ``RandomIoUCrop``,
#: ``SanitizeBoundingBoxes``, ``RandomHorizontalFlip``) using torchvision's
#: own implementations of the same named transforms rather than reimplementing
#: box-clipping/filtering math by hand. The recipe's own final ``Resize`` step
#: is left to the HF image processor at collate time (SPEC-010), which already
#: owns resizing; running it twice here would be redundant.
_AGGRESSIVE_TRAIN_TRANSFORM = tv_transforms.Compose(
    [
        tv_transforms.RandomPhotometricDistort(p=0.5),
        tv_transforms.RandomZoomOut(fill=0),
        tv_transforms.RandomApply([tv_transforms.RandomIoUCrop()], p=0.8),
        tv_transforms.SanitizeBoundingBoxes(min_size=1, labels_getter=_record_index_getter),
        tv_transforms.RandomHorizontalFlip(p=0.5),
    ]
)

#: The recipe's ``policy: stop_epoch: 71`` (out of 72): flip stays active in
#: the final epoch, the three more aggressive/geometric ops do not.
_FLIP_ONLY_TRAIN_TRANSFORM = tv_transforms.Compose([tv_transforms.RandomHorizontalFlip(p=0.5)])


class AugmentedFlirDataset(Dataset[tuple[Image.Image, dict[str, Any]]]):
    """Train-only wrapper applying SPEC-010's augmentation policy for one epoch.

    No hue/saturation perturbation carries physical meaning on FLIR thermal
    frames (single-channel data replicated into an RGB triplet, so every
    pixel already has zero saturation) — ``RandomPhotometricDistort``'s
    hue/saturation jitter is therefore a harmless no-op here, not a risk,
    so the full upstream transform is used unmodified rather than a
    thermal-specific subset. Randomness is derived per (epoch_seed, index)
    via a scoped ``torch.random.fork_rng`` + ``manual_seed`` (restored on
    exit, so global RNG state and other workers are unaffected) rather than
    hidden global state, keeping it reproducible and multi-worker-safe.
    """

    def __init__(
        self,
        base: Dataset[tuple[Image.Image, dict[str, Any]]],
        epoch_seed: int,
        *,
        aggressive: bool = True,
    ) -> None:
        self._base = base
        self._epoch_seed = epoch_seed
        self._transform = _AGGRESSIVE_TRAIN_TRANSFORM if aggressive else _FLIP_ONLY_TRAIN_TRANSFORM

    def __len__(self) -> int:
        return len(self._base)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[Image.Image, dict[str, Any]]:
        image, target = self._base[index]
        annotations = target["annotations"]
        width, height = image.size
        boxes_xyxy = torch.tensor(
            [
                [x, y, x + w, y + h]
                for x, y, w, h in (annotation["bbox"] for annotation in annotations)
            ],
            dtype=torch.float32,
        ).reshape(-1, 4)
        sample_target = {
            # torchvision's BoundingBoxes stub types __init__ against the
            # torch.Tensor overloads rather than its actual __new__ keyword
            # signature (format=/canvas_size=); verified working at runtime.
            "boxes": tv_tensors.BoundingBoxes(  # type: ignore[reportCallIssue]
                boxes_xyxy, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(height, width)
            ),
            "record_index": torch.arange(len(annotations), dtype=torch.long),
        }

        seed = derive_subseed(self._epoch_seed, "augment", str(index))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            image, transformed = self._transform(image, sample_target)

        new_annotations = []
        for box, original_index in zip(
            transformed["boxes"].tolist(), transformed["record_index"].tolist(), strict=True
        ):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            new_annotations.append(
                {**annotations[original_index], "bbox": [x1, y1, w, h], "area": w * h}
            )
        return image, {**target, "annotations": new_annotations}


def wrap_with_train_augmentation(
    base: Dataset[tuple[Image.Image, dict[str, Any]]], epoch_seed: int, *, aggressive: bool = True
) -> AugmentedFlirDataset:
    """Wrap a training dataset with SPEC-010's train-only augmentation for one epoch.

    ``aggressive=False`` drops to flip-only, matching the cited recipe's
    ``stop_epoch`` policy for the final training epoch.
    """
    return AugmentedFlirDataset(base, epoch_seed, aggressive=aggressive)
