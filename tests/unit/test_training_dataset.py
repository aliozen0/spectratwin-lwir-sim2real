import json
from pathlib import Path

import pytest
from PIL import Image

from spectratwin.real_data.adapter import scan_flir_dataset
from spectratwin.real_data.manifest import build_manifest
from spectratwin.real_data.records import FlirSampleRecord
from spectratwin.training.dataset import FLIR_ANNOTATION_FILENAME, load_training_dataset


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
