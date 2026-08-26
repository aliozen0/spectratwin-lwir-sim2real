import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import RTDetrImageProcessor, RTDetrV2Config, RTDetrV2ForObjectDetection

from spectratwin.real_data.manifest import build_manifest
from spectratwin.real_data.split import REAL_BENCHMARK, REAL_TRAIN
from spectratwin.training.dataset import FLIR_ANNOTATION_FILENAME, load_training_dataset
from spectratwin.training.evaluate import evaluate_frozen_benchmark

PROJECT_ID2LABEL = {0: "person", 1: "car", 2: "bicycle"}


def _write_flir_root(root: Path, n_images: int) -> None:
    categories = [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}]
    images = []
    annotations = []
    rng = np.random.default_rng(0)
    for i in range(n_images):
        file_name = f"img_{i}.jpg"
        images.append({"id": i, "file_name": file_name, "width": 100, "height": 100})
        annotations.append({"id": i, "image_id": i, "category_id": 1, "bbox": [10, 10, 20, 20]})
        pixels = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / file_name)
    (root / FLIR_ANNOTATION_FILENAME).write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories})
    )


def _build_benchmark_dataset(tmp_path: Path, n_images: int):
    root = tmp_path / "benchmark_root"
    root.mkdir()
    _write_flir_root(root, n_images)
    from spectratwin.real_data.adapter import scan_flir_dataset

    records = list(scan_flir_dataset(root, annotation_filename=FLIR_ANNOTATION_FILENAME).records)
    manifest = build_manifest(REAL_BENCHMARK, records, master_seed=0)
    dataset = load_training_dataset(manifest, root)
    return manifest, dataset


def _tiny_model(id2label: dict[int, str]):
    label2id = {v: k for k, v in id2label.items()}
    config = RTDetrV2Config(id2label=id2label, label2id=label2id, num_denoising=0)
    model = RTDetrV2ForObjectDetection(config)
    processor = RTDetrImageProcessor(size={"height": 160, "width": 160})
    return model, processor


def test_evaluate_frozen_benchmark_rejects_non_benchmark_manifest(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _write_flir_root(root, 1)
    from spectratwin.real_data.adapter import scan_flir_dataset

    records = list(scan_flir_dataset(root, annotation_filename=FLIR_ANNOTATION_FILENAME).records)
    wrong_manifest = build_manifest(REAL_TRAIN, records, master_seed=0)
    dataset = load_training_dataset(wrong_manifest, root)
    model, processor = _tiny_model(PROJECT_ID2LABEL)

    with pytest.raises(ValueError, match="real_benchmark"):
        evaluate_frozen_benchmark(
            model=model,
            processor=processor,
            manifest=wrong_manifest,
            dataset=dataset,
            device=torch.device("cpu"),
            artifact_root=tmp_path / "artifacts",
            run_id="run-1",
            checkpoint_id="test",
            checkpoint_revision="test",
            expected_id2label=PROJECT_ID2LABEL,
        )


def test_evaluate_frozen_benchmark_rejects_taxonomy_mismatch(tmp_path):
    manifest, dataset = _build_benchmark_dataset(tmp_path, 1)
    mismatched_model, processor = _tiny_model({0: "car", 1: "person", 2: "bicycle"})

    with pytest.raises(ValueError, match="taxonomy"):
        evaluate_frozen_benchmark(
            model=mismatched_model,
            processor=processor,
            manifest=manifest,
            dataset=dataset,
            device=torch.device("cpu"),
            artifact_root=tmp_path / "artifacts",
            run_id="run-1",
            checkpoint_id="test",
            checkpoint_revision="test",
            expected_id2label=PROJECT_ID2LABEL,
        )


def test_evaluate_frozen_benchmark_persists_report_and_predictions(tmp_path):
    manifest, dataset = _build_benchmark_dataset(tmp_path, 3)
    model, processor = _tiny_model(PROJECT_ID2LABEL)
    artifact_root = tmp_path / "artifacts"

    report = evaluate_frozen_benchmark(
        model=model,
        processor=processor,
        manifest=manifest,
        dataset=dataset,
        device=torch.device("cpu"),
        artifact_root=artifact_root,
        run_id="run-1",
        checkpoint_id="test-checkpoint",
        checkpoint_revision="rev-1",
        expected_id2label=PROJECT_ID2LABEL,
        seed=42,
    )

    assert report.benchmark_manifest_role == REAL_BENCHMARK
    assert report.benchmark_manifest_fingerprint == manifest.fingerprint
    assert report.metrics.sample_count == 3
    assert report.predictions_artifact_path.exists()
    assert report.failure_gallery_artifact_path.exists()
    assert (artifact_root / "eval" / "run-1" / "report.json").exists()

    predictions = json.loads(report.predictions_artifact_path.read_text())
    assert len(predictions) == 3
    assert {p["source_id"] for p in predictions} == {r.source_id for r in dataset.records}

    # Determinism: identical checkpoint/dataset -> identical metrics.
    report_again = evaluate_frozen_benchmark(
        model=model,
        processor=processor,
        manifest=manifest,
        dataset=dataset,
        device=torch.device("cpu"),
        artifact_root=artifact_root,
        run_id="run-2",
        checkpoint_id="test-checkpoint",
        checkpoint_revision="rev-1",
        expected_id2label=PROJECT_ID2LABEL,
        seed=42,
    )
    assert report_again.metrics == report.metrics
