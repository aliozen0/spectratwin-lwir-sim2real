"""COCO-style detection metrics for the RT-DETRv2 evaluation loop.

Wraps ``torchmetrics``' ``MeanAveragePrecision`` with its ``pycocotools``
backend, so results use the same IoU-matching/101-point interpolation
algorithm as the reference COCO evaluator (SPEC-010 acceptance: "a
meaningful FLIR baseline run produces COCO metrics").
"""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class DetectionMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    map: float
    map_50: float
    map_75: float
    map_small: float
    map_medium: float
    map_large: float
    map_per_class: dict[int, float]
    sample_count: int


def _target_to_torchmetrics(target: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Convert one ``FlirDetectionDataset`` target to xyxy boxes/labels."""
    boxes: list[list[float]] = []
    labels: list[int] = []
    for annotation in target["annotations"]:
        x, y, w, h = annotation["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(int(annotation["category_id"]))
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
        "labels": torch.tensor(labels, dtype=torch.int64)
        if labels
        else torch.zeros((0,), dtype=torch.int64),
    }


def accumulate_detection_metrics(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, Any]],
) -> DetectionMetrics:
    """Compute COCO AP metrics from already-produced predictions/targets.

    ``predictions`` entries are the ``boxes``/``scores``/``labels`` xyxy
    tensors an HF image processor's ``post_process_object_detection``
    returns. ``targets`` entries are the ``FlirDetectionDataset`` COCO-style
    per-image target shape (``bbox`` in ``xywh``, converted internally).
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions/targets length mismatch: {len(predictions)} != {len(targets)}"
        )
    if not predictions:
        raise ValueError("accumulate_detection_metrics requires at least one sample")

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(predictions, [_target_to_torchmetrics(target) for target in targets])
    computed = metric.compute()

    map_per_class: dict[int, float] = {}
    classes = computed.get("classes")
    per_class = computed.get("map_per_class")
    if classes is not None and per_class is not None:
        class_ids = classes.reshape(-1).tolist()
        class_values = per_class.reshape(-1).tolist()
        map_per_class = {
            int(class_id): float(value)
            for class_id, value in zip(class_ids, class_values, strict=True)
            if value >= 0.0  # torchmetrics reports -1 for classes absent from the batch
        }

    return DetectionMetrics(
        map=float(computed["map"]),
        map_50=float(computed["map_50"]),
        map_75=float(computed["map_75"]),
        # COCO convention: -1.0 means no object of that size was present.
        map_small=float(computed["map_small"]),
        map_medium=float(computed["map_medium"]),
        map_large=float(computed["map_large"]),
        map_per_class=map_per_class,
        sample_count=len(predictions),
    )


def evaluate_detection_dataset(
    *,
    model: Any,
    processor: Any,
    dataset: Any,
    device: torch.device,
    score_threshold: float = 0.0,
) -> DetectionMetrics:
    """Run inference over ``dataset`` and compute COCO AP metrics.

    ``dataset`` yields ``(PIL.Image, target)`` pairs in the
    ``FlirDetectionDataset`` shape. Predictions come from the model's own
    ``post_process_object_detection``, so metrics reflect the same
    score/box decoding a deployed run would use.
    """
    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, Any]] = []
    with torch.no_grad():
        for image, target in dataset:
            inputs = processor(images=[image], return_tensors="pt").to(device)
            outputs = model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            result = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=score_threshold
            )[0]
            predictions.append(
                {
                    "boxes": result["boxes"].detach().cpu(),
                    "scores": result["scores"].detach().cpu(),
                    "labels": result["labels"].detach().cpu(),
                }
            )
            targets.append(target)

    return accumulate_detection_metrics(predictions, targets)
