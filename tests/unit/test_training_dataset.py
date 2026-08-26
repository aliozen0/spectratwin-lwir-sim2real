import json
from pathlib import Path

import pytest
from PIL import Image

from spectratwin.real_data.adapter import scan_flir_dataset
from spectratwin.real_data.manifest import build_manifest
from spectratwin.real_data.records import FlirSampleRecord
from spectratwin.training.dataset import (
    FLIR_ANNOTATION_FILENAME,
    load_training_dataset,
    wrap_with_train_augmentation,
)


def _write_flir_root(root: Path, images: list[dict], annotations: list[dict]) -> None:
    categories = [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}, {"id": 3, "name": "bike"}]
    (root / FLIR_ANNOTATION_FILENAME).write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories})
    )
    for image in images:
        Image.new("RGB", (image["width"], image["height"])).save(root / image["file_name"])


def _tiny_root(tmp_path: Path) -> Path:
    images = [
        {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
        {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100},
    ]
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
        {"id": 2, "image_id": 2, "category_id": 2, "bbox": [5, 5, 10, 10]},
    ]
    _write_flir_root(tmp_path, images, annotations)
    return tmp_path


def _scan_records(root: Path) -> list[FlirSampleRecord]:
    result = scan_flir_dataset(root, annotation_filename=FLIR_ANNOTATION_FILENAME)
    assert result.issues == ()
    return list(result.records)


def test_load_training_dataset_matches_manifest(tmp_path):
    root = _tiny_root(tmp_path)
    records = _scan_records(root)
    manifest = build_manifest("real_train", records, master_seed=0)

    dataset = load_training_dataset(manifest, root)

    assert len(dataset) == 2
    image, target = dataset[0]
    assert image.size == (100, 100)
    assert target["annotations"][0]["category_id"] == 0  # person -> id 0


def test_load_training_dataset_filters_to_manifest_ids(tmp_path):
    root = _tiny_root(tmp_path)
    only_a = [record for record in _scan_records(root) if record.source_id == "a.jpg"]
    manifest = build_manifest("real_train", only_a, master_seed=0)

    dataset = load_training_dataset(manifest, root)

    assert len(dataset) == 1


def test_load_training_dataset_rejects_missing_sample_id(tmp_path):
    root = _tiny_root(tmp_path)
    phantom = [
        FlirSampleRecord(
            source_id="does-not-exist.jpg",
            sequence_key="x",
            relative_image_path="does-not-exist.jpg",
            width=100,
            height=100,
            image_sha256="0" * 64,
            annotations=(),
        )
    ]
    manifest = build_manifest("real_train", phantom, master_seed=0)

    with pytest.raises(ValueError, match="missing from a fresh scan"):
        load_training_dataset(manifest, root)


def test_load_training_dataset_rejects_tampered_fingerprint(tmp_path):
    root = _tiny_root(tmp_path)
    records = [record for record in _scan_records(root) if record.source_id == "a.jpg"]
    manifest = build_manifest("real_train", records, master_seed=0)
    # Tamper with the frozen fingerprint to simulate inconsistent manifest data.
    stale = manifest.model_copy(update={"fingerprint": "0" * 64})

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_training_dataset(stale, root)


def test_load_training_dataset_rejects_edited_annotation_content(tmp_path):
    root = _tiny_root(tmp_path)
    manifest = build_manifest("real_train", _scan_records(root), master_seed=0)
    annotation_path = root / FLIR_ANNOTATION_FILENAME
    payload = json.loads(annotation_path.read_text())
    payload["annotations"][0]["bbox"] = [11, 10, 20, 20]
    annotation_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_training_dataset(manifest, root)


def test_load_training_dataset_rejects_edited_image_bytes(tmp_path):
    root = _tiny_root(tmp_path)
    manifest = build_manifest("real_train", _scan_records(root), master_seed=0)
    Image.new("RGB", (100, 100), color="white").save(root / "a.jpg")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_training_dataset(manifest, root)


def test_load_training_dataset_ignores_annotation_json_formatting_changes(tmp_path):
    root = _tiny_root(tmp_path)
    manifest = build_manifest("real_train", _scan_records(root), master_seed=0)
    annotation_path = root / FLIR_ANNOTATION_FILENAME
    payload = json.loads(annotation_path.read_text())
    annotation_path.write_text(json.dumps(payload, indent=4, sort_keys=True))

    dataset = load_training_dataset(manifest, root)

    assert len(dataset) == 2


def test_dataset_getitem_rechecks_path_confinement_after_scan(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    _tiny_root(root)
    manifest = build_manifest("real_train", _scan_records(root), master_seed=0)
    dataset = load_training_dataset(manifest, root)
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (100, 100), color="white").save(outside)
    (root / "a.jpg").unlink()
    (root / "a.jpg").symlink_to(outside)

    with pytest.raises(ValueError, match="dataset root"):
        dataset[0]


def test_augmented_dataset_is_deterministic_per_epoch_seed(tmp_path):
    root = _tiny_root(tmp_path)
    manifest = build_manifest("real_train", _scan_records(root), master_seed=0)
    dataset = load_training_dataset(manifest, root)

    augmented_a = wrap_with_train_augmentation(dataset, epoch_seed=123)
    augmented_b = wrap_with_train_augmentation(dataset, epoch_seed=123)

    image_a, target_a = augmented_a[0]
    image_b, target_b = augmented_b[0]
    assert image_a.tobytes() == image_b.tobytes()
    assert target_a == target_b


def test_augmented_dataset_varies_across_epoch_seeds(tmp_path):
    # A jitter-only change is invisible on a solid-black fixture image (any
    # brightness/contrast scale of 0 is still 0), so this needs real pixel
    # variance to detect the jitter at all.
    images = [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}]
    annotations = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]}]
    _write_flir_root(tmp_path, images, annotations)
    Image.new("RGB", (100, 100), color=(120, 130, 140)).save(tmp_path / "a.jpg")
    manifest = build_manifest("real_train", _scan_records(tmp_path), master_seed=0)
    dataset = load_training_dataset(manifest, tmp_path)

    outcomes = {
        wrap_with_train_augmentation(dataset, epoch_seed=seed)[0][0].tobytes() for seed in range(10)
    }

    assert len(outcomes) > 1


def test_flip_only_pipeline_mirrors_bounding_box_and_preserves_size(tmp_path):
    # aggressive=False isolates flip from RandomZoomOut/RandomIoUCrop so the
    # exact mirrored coordinate can be asserted deterministically; the
    # aggressive pipeline can also change canvas size, which would make a
    # literal "width - x - w" check meaningless.
    root = _tiny_root(tmp_path)
    manifest = build_manifest("real_train", _scan_records(root), master_seed=0)
    dataset = load_training_dataset(manifest, root)
    base_image, base_target = dataset[0]
    width, height = base_image.size
    original_x, _y, original_w, _h = base_target["annotations"][0]["bbox"]

    flipped = next(
        (image, target)
        for seed in range(50)
        for image, target in [
            wrap_with_train_augmentation(dataset, epoch_seed=seed, aggressive=False)[0]
        ]
        if target["annotations"][0]["bbox"][0] != original_x
    )
    flipped_image, flipped_target = flipped

    assert flipped_image.size == (width, height)
    assert flipped_target["annotations"][0]["bbox"][0] == width - original_x - original_w
    assert (
        flipped_target["annotations"][0]["category_id"]
        == base_target["annotations"][0]["category_id"]
    )


def test_aggressive_pipeline_never_invents_boxes_and_stays_in_canvas(tmp_path):
    root = _tiny_root(tmp_path)
    manifest = build_manifest("real_train", _scan_records(root), master_seed=0)
    dataset = load_training_dataset(manifest, root)
    original_count = len(dataset[0][1]["annotations"])

    for seed in range(20):
        image, target = wrap_with_train_augmentation(dataset, epoch_seed=seed)[0]
        width, height = image.size
        assert len(target["annotations"]) <= original_count
        for annotation in target["annotations"]:
            x, y, w, h = annotation["bbox"]
            assert w > 0
            assert h > 0
            assert x >= -1e-3
            assert y >= -1e-3
            assert x + w <= width + 1e-3
            assert y + h <= height + 1e-3


def test_aggressive_pipeline_handles_zero_annotation_images(tmp_path):
    images = [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}]
    _write_flir_root(tmp_path, images, annotations=[])
    manifest = build_manifest("real_train", _scan_records(tmp_path), master_seed=0)
    dataset = load_training_dataset(manifest, tmp_path)

    image, target = wrap_with_train_augmentation(dataset, epoch_seed=0)[0]

    assert target["annotations"] == []
    assert image.size[0] > 0 and image.size[1] > 0
