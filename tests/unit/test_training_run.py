"""Offline smoke test for the real-only train/eval loop.

Uses a tiny randomly-initialized RT-DETRv2 model instead of the pretrained
HF checkpoint, so this test needs no network access and stays fast - the
real pretrained-checkpoint path is exercised by a separate local smoke run,
not by CI.
"""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spectratwin.config.settings import ExecutionProfile, Settings
from spectratwin.real_data.adapter import scan_flir_dataset
from spectratwin.real_data.manifest import build_manifest, write_manifest
from spectratwin.real_data.records import FlirSampleRecord
from spectratwin.training.config import RealSmokeTrainConfig
from spectratwin.training.dataset import FLIR_ANNOTATION_FILENAME
from spectratwin.training.run import run_real_smoke_training


def _write_flir_root(root: Path, n_images: int) -> list[FlirSampleRecord]:
    categories = [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}]
    images = []
    annotations = []
    # Random pixel content, not a solid color: a zero-variance image drives
    # the untrained model's backward pass to NaN gradients (reproduced and
    # confirmed independent of this project's training-loop code).
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
    result = scan_flir_dataset(root, annotation_filename=FLIR_ANNOTATION_FILENAME)
    assert result.issues == ()
    return list(result.records)


def _tiny_model_factory():
    from transformers import RTDetrImageProcessor, RTDetrV2Config, RTDetrV2ForObjectDetection

    def factory(_checkpoint_id: str, _checkpoint_revision: str):
        config = RTDetrV2Config(
            id2label={0: "person", 1: "car", 2: "bicycle"},
            label2id={"person": 0, "car": 1, "bicycle": 2},
            # Denoising is a training-refinement trick, not required to prove
            # a train/eval step executes; disabled here because it produces
            # NaN losses on a randomly-initialized (untrained) test model.
            num_denoising=0,
        )
        model = RTDetrV2ForObjectDetection(config)
        processor = RTDetrImageProcessor(size={"height": 160, "width": 160})
        return model, processor

    return factory


def test_run_real_smoke_training_end_to_end(tmp_path):
    train_root = tmp_path / "train_root"
    dev_root = tmp_path / "dev_root"
    train_root.mkdir()
    dev_root.mkdir()

    train_records = _write_flir_root(train_root, 3)
    dev_records = _write_flir_root(dev_root, 2)

    train_manifest = build_manifest("real_train", train_records, master_seed=0)
    dev_manifest = build_manifest("real_dev", dev_records, master_seed=0)

    train_manifest_path = tmp_path / "real_train.json"
    dev_manifest_path = tmp_path / "real_dev.json"
    write_manifest(train_manifest, train_manifest_path)
    write_manifest(dev_manifest, dev_manifest_path)

    # Not named "artifacts": MLflow 3.9.0's FileStore fails to create a run
    # (raises "Run '<id>' not found" from within create_run) when any path
    # component in the tracking-uri ancestry is literally "artifacts" -
    # reproduced independently of this project's code. See run.py.
    artifact_root = tmp_path / "run-artifacts"
    settings = Settings(
        execution_profile=ExecutionProfile.WSL,
        master_seed=0,
        data_root=tmp_path,
        cache_root=tmp_path,
        artifact_root=artifact_root,
    )

    config = RealSmokeTrainConfig(
        train_manifest_path=train_manifest_path,
        dev_manifest_path=dev_manifest_path,
        flir_train_root=train_root,
        flir_dev_root=dev_root,
        max_train_samples=2,
        max_dev_samples=1,
        max_steps=1,
        batch_size=1,
        device="cpu",
    )

    result = run_real_smoke_training(config, settings, model_factory=_tiny_model_factory())

    assert result.checkpoint_path.exists()
    assert result.checkpoint_path.parent.parent == artifact_root / "checkpoints"
    assert result.train_dataset_fingerprint == train_manifest.fingerprint
    assert result.dev_dataset_fingerprint == dev_manifest.fingerprint
    assert result.seed >= 0
    assert math.isfinite(result.final_train_loss)
    assert math.isfinite(result.final_eval_loss)
    assert result.train_sample_count == 2
    assert result.dev_sample_count == 1
    assert result.resumed_from_step == 0
    assert result.completed_steps == 1

    mlruns_dir = artifact_root / "mlruns"
    assert mlruns_dir.exists()
    run_dirs = list(mlruns_dir.glob("*/" + result.mlflow_run_id))
    assert len(run_dirs) == 1


def test_run_real_smoke_resumes_from_persisted_checkpoint(tmp_path):
    train_root = tmp_path / "train_root"
    dev_root = tmp_path / "dev_root"
    train_root.mkdir()
    dev_root.mkdir()
    train_records = _write_flir_root(train_root, 3)
    dev_records = _write_flir_root(dev_root, 2)
    train_manifest = build_manifest("real_train", train_records, master_seed=0)
    dev_manifest = build_manifest("real_dev", dev_records, master_seed=0)
    train_manifest_path = tmp_path / "real_train.json"
    dev_manifest_path = tmp_path / "real_dev.json"
    write_manifest(train_manifest, train_manifest_path)
    write_manifest(dev_manifest, dev_manifest_path)

    first_artifact_root = tmp_path / "first-run-artifacts"
    first_settings = Settings(
        execution_profile=ExecutionProfile.WSL,
        master_seed=0,
        data_root=tmp_path,
        cache_root=tmp_path,
        artifact_root=first_artifact_root,
    )
    common_config = {
        "train_manifest_path": train_manifest_path,
        "dev_manifest_path": dev_manifest_path,
        "flir_train_root": train_root,
        "flir_dev_root": dev_root,
        "max_train_samples": 2,
        "max_dev_samples": 1,
        "batch_size": 1,
        "device": "cpu",
        "run_id": "resume-smoke",
    }
    first_result = run_real_smoke_training(
        RealSmokeTrainConfig(**common_config, max_steps=1),
        first_settings,
        model_factory=_tiny_model_factory(),
    )

    persisted_checkpoint = tmp_path / "persistent" / "model.pt"
    persisted_checkpoint.parent.mkdir()
    shutil.copy2(first_result.checkpoint_path, persisted_checkpoint)

    resumed_artifact_root = tmp_path / "resumed-run-artifacts"
    resumed_settings = first_settings.model_copy(update={"artifact_root": resumed_artifact_root})
    resumed_result = run_real_smoke_training(
        RealSmokeTrainConfig(
            **common_config,
            max_steps=2,
            resume_from_checkpoint=persisted_checkpoint,
        ),
        resumed_settings,
        model_factory=_tiny_model_factory(),
    )

    assert resumed_result.run_id == "resume-smoke"
    assert resumed_result.resumed_from_step == 1
    assert resumed_result.completed_steps == 2
    bundle = __import__("torch").load(
        resumed_result.checkpoint_path, map_location="cpu", weights_only=True
    )
    assert bundle["schema_version"] == "spectratwin-training-checkpoint-v1"
    assert bundle["completed_steps"] == 2


def test_resume_rejects_changed_optimization_identity(tmp_path):
    train_root = tmp_path / "train_root"
    dev_root = tmp_path / "dev_root"
    train_root.mkdir()
    dev_root.mkdir()
    train_records = _write_flir_root(train_root, 2)
    dev_records = _write_flir_root(dev_root, 1)
    train_manifest = build_manifest("real_train", train_records, master_seed=0)
    dev_manifest = build_manifest("real_dev", dev_records, master_seed=0)
    train_manifest_path = tmp_path / "real_train.json"
    dev_manifest_path = tmp_path / "real_dev.json"
    write_manifest(train_manifest, train_manifest_path)
    write_manifest(dev_manifest, dev_manifest_path)
    settings = Settings(
        execution_profile=ExecutionProfile.WSL,
        master_seed=0,
        data_root=tmp_path,
        cache_root=tmp_path,
        artifact_root=tmp_path / "run-artifacts",
    )
    base_config = {
        "train_manifest_path": train_manifest_path,
        "dev_manifest_path": dev_manifest_path,
        "flir_train_root": train_root,
        "flir_dev_root": dev_root,
        "max_train_samples": 1,
        "max_dev_samples": 1,
        "max_steps": 1,
        "batch_size": 1,
        "device": "cpu",
        "run_id": "identity-smoke",
    }
    first_result = run_real_smoke_training(
        RealSmokeTrainConfig(**base_config), settings, model_factory=_tiny_model_factory()
    )

    with pytest.raises(ValueError, match="config_identity"):
        run_real_smoke_training(
            RealSmokeTrainConfig(
                **base_config,
                learning_rate=2e-4,
                resume_from_checkpoint=first_result.checkpoint_path,
            ),
            settings.model_copy(update={"artifact_root": tmp_path / "second-artifacts"}),
            model_factory=_tiny_model_factory(),
        )


def test_run_id_rejects_path_traversal():
    with pytest.raises(ValueError):
        RealSmokeTrainConfig(
            train_manifest_path=Path("train.json"),
            dev_manifest_path=Path("dev.json"),
            flir_train_root=Path("train"),
            flir_dev_root=Path("dev"),
            run_id="../outside",
        )
