import pytest
import torch

from spectratwin.training.metrics import accumulate_detection_metrics


def _target(image_id: int, boxes_xywh: list[list[float]], category_ids: list[int]) -> dict:
    return {
        "image_id": image_id,
        "annotations": [
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "id": i,
            }
            for i, (bbox, category_id) in enumerate(zip(boxes_xywh, category_ids, strict=True))
        ],
    }


def test_accumulate_detection_metrics_perfect_prediction_scores_map_one() -> None:
    target = _target(0, [[10.0, 10.0, 20.0, 20.0]], [0])
    prediction = {
        "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
        "scores": torch.tensor([0.99]),
        "labels": torch.tensor([0]),
    }

    metrics = accumulate_detection_metrics([prediction], [target])

    assert metrics.map == pytest.approx(1.0)
    assert metrics.map_50 == pytest.approx(1.0)
    assert metrics.map_75 == pytest.approx(1.0)
    assert metrics.map_per_class == {0: pytest.approx(1.0)}
    assert metrics.sample_count == 1


def test_accumulate_detection_metrics_no_predictions_scores_map_zero() -> None:
    target = _target(0, [[10.0, 10.0, 20.0, 20.0]], [0])
    prediction = {
        "boxes": torch.zeros((0, 4)),
        "scores": torch.zeros((0,)),
        "labels": torch.zeros((0,), dtype=torch.int64),
    }

    metrics = accumulate_detection_metrics([prediction], [target])

    assert metrics.map == pytest.approx(0.0)
    assert metrics.sample_count == 1


def test_accumulate_detection_metrics_empty_input_target_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        accumulate_detection_metrics(
            [
                {
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }
            ],
            [],
        )


def test_accumulate_detection_metrics_empty_batch_raises() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        accumulate_detection_metrics([], [])
