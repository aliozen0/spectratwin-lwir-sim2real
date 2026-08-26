"""Frozen real-benchmark evaluation report (SPEC-011).

Runs a checkpoint against the frozen ``real_benchmark`` manifest exactly
once per report and persists everything needed to reproduce or audit it:
COCO metrics, a per-sample prediction artifact, a bounded false-positive/
false-negative gallery, and run/dataset/model provenance. Read-only:
nothing here writes back to the benchmark manifest or FLIR source root.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict
from torchvision.ops import box_iou

from spectratwin.real_data.manifest import DatasetManifest
from spectratwin.real_data.split import REAL_BENCHMARK
from spectratwin.training.dataset import FlirDetectionDataset
from spectratwin.training.metrics import DetectionMetrics, accumulate_detection_metrics

#: COCO matching convention this gallery follows for both false positives
#: and false negatives.
MATCH_IOU_THRESHOLD = 0.5

#: Cap each failure category so the gallery stays a diagnostic sample, not
#: a full dump of every miss on a large benchmark.
MAX_GALLERY_EXAMPLES_PER_KIND = 50


class FailureExample(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    kind: str  # "false_positive" | "false_negative"
    category_id: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float | None = None


class PredictionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    predicted_boxes_xyxy: list[tuple[float, float, float, float]]
    predicted_labels: list[int]
    predicted_scores: list[float]
    ground_truth_boxes_xyxy: list[tuple[float, float, float, float]]
    ground_truth_labels: list[int]


class EvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    git_sha: str | None
    checkpoint_id: str
    checkpoint_revision: str
    benchmark_manifest_role: str
    benchmark_manifest_fingerprint: str
    seed: int | None
    metrics: DetectionMetrics
    false_positive_count: int
    false_negative_count: int
    predictions_artifact_path: Path
    failure_gallery_artifact_path: Path


def _assert_taxonomy_matches(model: Any, expected_id2label: dict[int, str]) -> None:
    actual = {int(k): v for k, v in dict(model.config.id2label).items()}
    if actual != expected_id2label:
        raise ValueError(
            "checkpoint taxonomy does not match the project taxonomy: "
            f"expected {expected_id2label}, got {actual}"
        )


def _bbox4(row: torch.Tensor) -> tuple[float, float, float, float]:
    values = row.tolist()
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _target_boxes_xyxy(target: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    for annotation in target["annotations"]:
        x, y, w, h = annotation["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(int(annotation["category_id"]))
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
    labels_tensor = (
        torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
    )
    return boxes_tensor, labels_tensor


def _match_failures(
    *,
    source_id: str,
    predicted_boxes: torch.Tensor,
    predicted_labels: torch.Tensor,
    predicted_scores: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
) -> list[FailureExample]:
    """Greedy same-class IoU matching; unmatched entries are the gallery."""
    matched_targets: set[int] = set()
    matched_predictions: set[int] = set()

    if len(predicted_boxes) and len(target_boxes):
        iou = box_iou(predicted_boxes, target_boxes)
        # Highest-IoU predictions get first pick, matching COCO's greedy
        # detection-matching behavior instead of an unordered scan.
        order = torch.argsort(predicted_scores, descending=True).tolist()
        for pred_index in order:
            best_target_index: int | None = None
            best_iou = MATCH_IOU_THRESHOLD
            for target_index in range(len(target_boxes)):
                if target_index in matched_targets:
                    continue
                if int(predicted_labels[pred_index]) != int(target_labels[target_index]):
                    continue
                candidate_iou = float(iou[pred_index, target_index])
                if candidate_iou >= best_iou:
                    best_iou = candidate_iou
                    best_target_index = target_index
            if best_target_index is not None:
                matched_targets.add(best_target_index)
                matched_predictions.add(pred_index)

    examples: list[FailureExample] = []
    for pred_index in range(len(predicted_boxes)):
        if pred_index in matched_predictions:
            continue
        examples.append(
            FailureExample(
                source_id=source_id,
                kind="false_positive",
                category_id=int(predicted_labels[pred_index]),
                bbox_xyxy=_bbox4(predicted_boxes[pred_index]),
                score=float(predicted_scores[pred_index]),
            )
        )
    for target_index in range(len(target_boxes)):
        if target_index in matched_targets:
            continue
        examples.append(
            FailureExample(
                source_id=source_id,
                kind="false_negative",
                category_id=int(target_labels[target_index]),
                bbox_xyxy=_bbox4(target_boxes[target_index]),
                score=None,
            )
        )
    return examples


def evaluate_frozen_benchmark(
    *,
    model: Any,
    processor: Any,
    manifest: DatasetManifest,
    dataset: FlirDetectionDataset,
    device: torch.device,
    artifact_root: Path,
    run_id: str,
    checkpoint_id: str,
    checkpoint_revision: str,
    expected_id2label: dict[int, str],
    git_sha: str | None = None,
    seed: int | None = None,
    score_threshold: float = 0.0,
) -> EvaluationReport:
    """Evaluate ``dataset`` (built from the frozen benchmark manifest) once.

    Rejects a checkpoint whose taxonomy does not match the project taxonomy
    before running any inference. ``dataset.records`` must be in the same
    order ``dataset`` indexes so predictions can be traced back to a
    ``source_id``.
    """
    if manifest.role != REAL_BENCHMARK:
        raise ValueError(
            f"evaluate_frozen_benchmark requires role={REAL_BENCHMARK!r}, got {manifest.role!r}"
        )
    if len(dataset) != len(dataset.records):
        raise ValueError("dataset/records length mismatch; cannot trace predictions to source_id")

    _assert_taxonomy_matches(model, expected_id2label)

    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, Any]] = []
    prediction_records: list[PredictionRecord] = []
    failures: list[FailureExample] = []

    with torch.no_grad():
        for index in range(len(dataset)):
            image, target = dataset[index]
            record = dataset.records[index]
            inputs = processor(images=[image], return_tensors="pt").to(device)
            outputs = model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            result = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=score_threshold
            )[0]
            predicted_boxes = result["boxes"].detach().cpu()
            predicted_labels = result["labels"].detach().cpu()
            predicted_scores = result["scores"].detach().cpu()

            predictions.append(
                {"boxes": predicted_boxes, "scores": predicted_scores, "labels": predicted_labels}
            )
            targets.append(target)

            target_boxes, target_labels = _target_boxes_xyxy(target)
            prediction_records.append(
                PredictionRecord(
                    source_id=record.source_id,
                    predicted_boxes_xyxy=[_bbox4(row) for row in predicted_boxes],
                    predicted_labels=[int(v) for v in predicted_labels.tolist()],
                    predicted_scores=[float(v) for v in predicted_scores.tolist()],
                    ground_truth_boxes_xyxy=[_bbox4(row) for row in target_boxes],
                    ground_truth_labels=[int(v) for v in target_labels.tolist()],
                )
            )
            failures.extend(
                _match_failures(
                    source_id=record.source_id,
                    predicted_boxes=predicted_boxes,
                    predicted_labels=predicted_labels,
                    predicted_scores=predicted_scores,
                    target_boxes=target_boxes,
                    target_labels=target_labels,
                )
            )

    metrics = accumulate_detection_metrics(predictions, targets)

    false_positive_count = sum(1 for f in failures if f.kind == "false_positive")
    false_negative_count = sum(1 for f in failures if f.kind == "false_negative")
    gallery = [f for f in failures if f.kind == "false_positive"][
        :MAX_GALLERY_EXAMPLES_PER_KIND
    ] + [f for f in failures if f.kind == "false_negative"][:MAX_GALLERY_EXAMPLES_PER_KIND]

    eval_dir = artifact_root / "eval" / run_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = eval_dir / "predictions.json"
    gallery_path = eval_dir / "failure_gallery.json"
    predictions_path.write_text(
        json.dumps([p.model_dump(mode="json") for p in prediction_records], indent=2)
    )
    gallery_path.write_text(json.dumps([f.model_dump(mode="json") for f in gallery], indent=2))

    report = EvaluationReport(
        run_id=run_id,
        git_sha=git_sha,
        checkpoint_id=checkpoint_id,
        checkpoint_revision=checkpoint_revision,
        benchmark_manifest_role=manifest.role,
        benchmark_manifest_fingerprint=manifest.fingerprint,
        seed=seed,
        metrics=metrics,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        predictions_artifact_path=predictions_path,
        failure_gallery_artifact_path=gallery_path,
    )
    (eval_dir / "report.json").write_text(report.model_dump_json(indent=2))
    return report
