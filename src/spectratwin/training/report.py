"""Diagnostic plots and tables for a training/evaluation run.

Reads only what a run already logged (MLflow metrics, an
:class:`~spectratwin.training.evaluate.EvaluationReport` predictions
artifact); it never runs training or evaluation itself and never reads
``real_benchmark``. Everything here is a rendering of already-computed
numbers, not a new measurement.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

# Must happen before ``import matplotlib``: matplotlib reads MPLBACKEND at
# import time (e.g. Colab/Jupyter sets it to an inline backend that isn't
# valid outside a notebook kernel), so matplotlib.use("Agg") after the
# import is already too late to prevent that crash.
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import torch
from mlflow.tracking import MlflowClient
from torchvision.ops import box_iou

from spectratwin.training.model import ID2LABEL

#: Loss/metric names plotted together on the training-report loss curve, in
#: the order they should appear if present.
_LOSS_CURVE_METRICS: tuple[str, ...] = ("train_loss", "epoch_train_loss", "dev_loss", "eval_loss")

#: Confusion-matrix bin standing in for "no matching box" (a false positive's
#: predicted row or a false negative's ground-truth column).
_BACKGROUND_LABEL = "background"


def _find_run(tracking_uri: str, experiment_name: str, run_id: str):
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"no MLflow experiment named {experiment_name!r} at {tracking_uri!r}")
    runs = client.search_runs(
        [experiment.experiment_id], filter_string=f"tags.mlflow.runName = '{run_id}'"
    )
    if not runs:
        raise ValueError(f"no MLflow run named {run_id!r} in experiment {experiment_name!r}")
    return client, runs[0]


def build_training_loss_report(
    tracking_uri: str,
    experiment_name: str,
    run_id: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Render a loss-curve plot and per-metric CSV tables for one MLflow run.

    Writes ``loss_curve.png``, one ``<metric>.csv`` per logged metric and a
    ``summary.json`` (run params/final metrics) under ``output_dir``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    client, run = _find_run(tracking_uri, experiment_name, run_id)

    metric_names = sorted(run.data.metrics.keys())
    histories = {name: client.get_metric_history(run.info.run_id, name) for name in metric_names}

    written: dict[str, Path] = {}
    for name, points in histories.items():
        csv_path = output_dir / f"{name}.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["step", "value", "timestamp_ms"])
            for point in sorted(points, key=lambda p: p.step):
                writer.writerow([point.step, point.value, point.timestamp])
        written[f"metric_csv:{name}"] = csv_path

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_any = False
    for name in _LOSS_CURVE_METRICS:
        points = histories.get(name)
        if not points:
            continue
        ordered = sorted(points, key=lambda p: p.step)
        ax.plot(
            [p.step for p in ordered],
            [p.value for p in ordered],
            marker="o",
            markersize=3,
            label=name,
        )
        plotted_any = True
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"Training/dev loss — run {run_id}")
    if plotted_any:
        ax.legend()
    ax.grid(True, alpha=0.3)
    loss_curve_path = output_dir / "loss_curve.png"
    fig.savefig(loss_curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written["loss_curve_png"] = loss_curve_path

    summary = {
        "run_id": run_id,
        "mlflow_run_id": run.info.run_id,
        "experiment_name": experiment_name,
        "params": dict(run.data.params),
        "final_metrics": dict(run.data.metrics),
        "note": (
            "Execution diagnostics read back from a training run's own MLflow "
            "log. Not a benchmark evaluation; real_benchmark is never read here."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    written["summary_json"] = summary_path

    return written


def _match_confusion_counts(
    predictions: list[dict],
    id2label: dict[int, str],
    iou_threshold: float,
    score_threshold: float,
) -> dict[str, dict[str, int]]:
    """Greedy IoU-matched confusion counts across every prediction record.

    Each ground-truth box is matched to its highest-IoU unmatched prediction
    (at or above ``score_threshold``); an unmatched ground truth counts as
    ``background`` predicted, an unmatched high-score prediction counts as a
    ``background`` ground truth (a false positive).
    """
    labels = [id2label[i] for i in sorted(id2label)] + [_BACKGROUND_LABEL]
    counts: dict[str, dict[str, int]] = {gt: dict.fromkeys(labels, 0) for gt in labels}

    for record in predictions:
        gt_boxes = torch.tensor(record["ground_truth_boxes_xyxy"], dtype=torch.float32)
        gt_labels = record["ground_truth_labels"]
        kept = [
            (box, label)
            for box, label, score in zip(
                record["predicted_boxes_xyxy"],
                record["predicted_labels"],
                record["predicted_scores"],
                strict=True,
            )
            if score >= score_threshold
        ]
        pred_boxes = torch.tensor([box for box, _ in kept], dtype=torch.float32)
        pred_labels = [label for _, label in kept]

        matched_pred = set()
        if gt_boxes.numel() and pred_boxes.numel():
            ious = box_iou(gt_boxes, pred_boxes)
        else:
            ious = torch.zeros((gt_boxes.shape[0], pred_boxes.shape[0]))

        for gt_index, gt_label in enumerate(gt_labels):
            best_iou = 0.0
            best_pred = -1
            for pred_index in range(pred_boxes.shape[0]):
                if pred_index in matched_pred:
                    continue
                iou = float(ious[gt_index, pred_index])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_pred = pred_index
            gt_name = id2label[int(gt_label)]
            if best_pred >= 0:
                matched_pred.add(best_pred)
                counts[gt_name][id2label[int(pred_labels[best_pred])]] += 1
            else:
                counts[gt_name][_BACKGROUND_LABEL] += 1

        for pred_index, pred_label in enumerate(pred_labels):
            if pred_index not in matched_pred:
                counts[_BACKGROUND_LABEL][id2label[int(pred_label)]] += 1

    return counts


def build_confusion_matrix_report(
    predictions_artifact_path: Path,
    output_dir: Path,
    id2label: dict[int, str] | None = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> dict[str, Path]:
    """Render a confusion-matrix heatmap and CSV from a saved predictions artifact.

    ``predictions_artifact_path`` is the ``predictions.json`` an
    :func:`~spectratwin.training.evaluate.evaluate_real_baseline_checkpoint`
    call already wrote; this only visualizes it.
    """
    id2label = id2label or ID2LABEL
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = json.loads(predictions_artifact_path.read_text())

    counts = _match_confusion_counts(predictions, id2label, iou_threshold, score_threshold)
    labels = [id2label[i] for i in sorted(id2label)] + [_BACKGROUND_LABEL]

    csv_path = output_dir / "confusion_matrix.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ground_truth \\ predicted", *labels])
        for gt_label in labels:
            writer.writerow([gt_label, *(counts[gt_label][pred] for pred in labels)])

    matrix = [[counts[gt][pred] for pred in labels] for gt in labels]
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2, 1.4 * len(labels) + 2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(f"Confusion matrix (IoU≥{iou_threshold}, score≥{score_threshold})")
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, str(matrix[row][col]), ha="center", va="center")
    fig.colorbar(image, ax=ax, shrink=0.8)
    png_path = output_dir / "confusion_matrix.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"confusion_matrix_csv": csv_path, "confusion_matrix_png": png_path}


def build_per_class_ap_report(
    metrics_map_per_class: dict[int, float],
    output_dir: Path,
    id2label: dict[int, str] | None = None,
) -> dict[str, Path]:
    """Render a per-class average-precision bar chart from a computed report's metrics."""
    id2label = id2label or ID2LABEL
    output_dir.mkdir(parents=True, exist_ok=True)

    class_ids = sorted(metrics_map_per_class)
    names = [id2label.get(i, str(i)) for i in class_ids]
    values = [metrics_map_per_class[i] for i in class_ids]

    fig, ax = plt.subplots(figsize=(max(4, len(names) * 1.2), 4))
    ax.bar(names, values)
    ax.set_ylabel("AP")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class average precision")
    ax.grid(True, axis="y", alpha=0.3)
    png_path = output_dir / "per_class_ap.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_dir / "per_class_ap.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["category", "ap"])
        for name, value in zip(names, values, strict=True):
            writer.writerow([name, value])

    return {"per_class_ap_png": png_path, "per_class_ap_csv": csv_path}
