import json
from pathlib import Path

import mlflow

from spectratwin.training.report import (
    build_confusion_matrix_report,
    build_per_class_ap_report,
    build_training_loss_report,
)


def test_build_training_loss_report_writes_curve_tables_and_summary(tmp_path: Path) -> None:
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-experiment")
    with mlflow.start_run(run_name="test-run"):
        mlflow.log_params({"epochs": 1})
        for step, value in enumerate([1.0, 0.8, 0.6]):
            mlflow.log_metric("train_loss", value, step=step)
        mlflow.log_metric("dev_loss", 0.7, step=1)

    output_dir = tmp_path / "report"
    written = build_training_loss_report(
        tracking_uri=tracking_uri,
        experiment_name="test-experiment",
        run_id="test-run",
        output_dir=output_dir,
    )

    assert written["loss_curve_png"].is_file()
    train_loss_csv = written["metric_csv:train_loss"]
    rows = train_loss_csv.read_text().strip().splitlines()
    assert rows[0] == "step,value,timestamp_ms"
    assert len(rows) == 4  # header + 3 points

    summary = json.loads(written["summary_json"].read_text())
    assert summary["run_id"] == "test-run"
    assert summary["final_metrics"]["train_loss"] == 0.6
    assert "real_benchmark" not in summary["note"] or "never" in summary["note"]


def test_build_training_loss_report_merges_a_crash_and_resume(tmp_path: Path) -> None:
    """A resumed baseline run opens a second physical MLflow run under the
    same run_name; the report must stitch both histories into one curve
    instead of only showing the resumed tail."""
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-experiment")
    with mlflow.start_run(run_name="crash-resume-run"):
        for step, value in enumerate([1.0, 0.9, 0.8]):
            mlflow.log_metric("train_loss", value, step=step)
    with mlflow.start_run(run_name="crash-resume-run"):
        for step, value in [(3, 0.7), (4, 0.6)]:
            mlflow.log_metric("train_loss", value, step=step)
        mlflow.log_metric("dev_loss", 0.65, step=1)

    written = build_training_loss_report(
        tracking_uri=tracking_uri,
        experiment_name="test-experiment",
        run_id="crash-resume-run",
        output_dir=tmp_path / "report",
    )

    rows = written["metric_csv:train_loss"].read_text().strip().splitlines()[1:]
    steps = [int(row.split(",")[0]) for row in rows]
    values = [float(row.split(",")[1]) for row in rows]
    assert steps == [0, 1, 2, 3, 4]
    assert values == [1.0, 0.9, 0.8, 0.7, 0.6]

    summary = json.loads(written["summary_json"].read_text())
    assert len(summary["mlflow_run_ids"]) == 2


def _prediction_record(source_id, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
    return {
        "source_id": source_id,
        "predicted_boxes_xyxy": pred_boxes,
        "predicted_labels": pred_labels,
        "predicted_scores": pred_scores,
        "ground_truth_boxes_xyxy": gt_boxes,
        "ground_truth_labels": gt_labels,
    }


def test_build_confusion_matrix_report_counts_matches_and_misses(tmp_path: Path) -> None:
    id2label = {0: "person", 1: "car"}
    predictions = [
        # Correct person match.
        _prediction_record("img-0", [[0, 0, 10, 10]], [0], [0.9], [[0, 0, 10, 10]], [0]),
        # False negative: ground-truth car with no matching prediction.
        _prediction_record("img-1", [], [], [], [[0, 0, 10, 10]], [1]),
        # False positive: high-score prediction with no ground truth.
        _prediction_record("img-2", [[0, 0, 10, 10]], [0], [0.9], [], []),
    ]
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(json.dumps(predictions))

    written = build_confusion_matrix_report(
        predictions_artifact_path=predictions_path,
        output_dir=tmp_path / "report",
        id2label=id2label,
    )

    assert written["confusion_matrix_png"].is_file()
    rows = written["confusion_matrix_csv"].read_text().strip().splitlines()
    header = rows[0].split(",")
    assert header == ["ground_truth \\ predicted", "person", "car", "background"]
    body = {row.split(",")[0]: [int(v) for v in row.split(",")[1:]] for row in rows[1:]}
    assert body["person"] == [1, 0, 0]  # one correct person match
    assert body["car"] == [0, 0, 1]  # unmatched car -> background
    assert body["background"] == [1, 0, 0]  # unmatched person prediction -> false positive


def test_build_per_class_ap_report_writes_bar_chart_and_csv(tmp_path: Path) -> None:
    written = build_per_class_ap_report(
        metrics_map_per_class={0: 0.5, 1: 0.75},
        output_dir=tmp_path / "report",
        id2label={0: "person", 1: "car"},
    )

    assert written["per_class_ap_png"].is_file()
    rows = written["per_class_ap_csv"].read_text().strip().splitlines()
    assert rows[0] == "category,ap"
    assert rows[1] == "person,0.5"
    assert rows[2] == "car,0.75"
