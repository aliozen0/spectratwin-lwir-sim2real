import hashlib
import json
from pathlib import Path

import pytest

from spectratwin.real_data.adapter import scan_flir_dataset
from spectratwin.real_data.records import ScanErrorCategory


def _write_annotations(
    root: Path, images: list[dict], annotations: list[dict], categories: list[dict]
):
    (root / "thermal_annotations.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories})
    )


def _touch_image(root: Path, name: str) -> None:
    (root / name).write_bytes(b"\x00")


CATEGORIES = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "car"},
    {"id": 3, "name": "bike"},
    {"id": 4, "name": "dog"},
]


def test_scan_normalizes_valid_subset(tmp_path):
    images = [
        {"id": 1, "file_name": "clip_001.jpg", "width": 100, "height": 100},
        {"id": 2, "file_name": "clip_002.jpg", "width": 100, "height": 100},
    ]
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
        {"id": 2, "image_id": 2, "category_id": 3, "bbox": [5, 5, 10, 10]},
    ]
    _write_annotations(tmp_path, images, annotations, CATEGORIES)
    _touch_image(tmp_path, "clip_001.jpg")
    _touch_image(tmp_path, "clip_002.jpg")

    result = scan_flir_dataset(tmp_path)

    assert len(result.records) == 2
    assert result.issues == ()
    by_id = {r.source_id: r for r in result.records}
    assert by_id["clip_001.jpg"].annotations[0].project_category == "person"
    assert by_id["clip_002.jpg"].annotations[0].project_category == "bicycle"
    assert by_id["clip_001.jpg"].image_sha256 == hashlib.sha256(b"\x00").hexdigest()


def test_scan_reports_missing_annotation_file(tmp_path):
    result = scan_flir_dataset(tmp_path)
    assert result.records == ()
    assert result.issues[0].category == ScanErrorCategory.MISSING_ANNOTATION_FILE


def test_scan_reports_corrupt_annotation_file(tmp_path):
    (tmp_path / "thermal_annotations.json").write_text("{not json")
    result = scan_flir_dataset(tmp_path)
    assert result.issues[0].category == ScanErrorCategory.CORRUPT_ANNOTATION_FILE


def test_scan_reports_missing_image(tmp_path):
    images = [{"id": 1, "file_name": "missing.jpg", "width": 100, "height": 100}]
    _write_annotations(tmp_path, images, [], CATEGORIES)

    result = scan_flir_dataset(tmp_path)

    assert result.records == ()
    assert result.issues[0].category == ScanErrorCategory.MISSING_IMAGE


def test_scan_rejects_sample_with_invalid_project_bbox(tmp_path):
    images = [{"id": 1, "file_name": "clip_001.jpg", "width": 100, "height": 100}]
    annotations = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [90, 90, 50, 50]}]
    _write_annotations(tmp_path, images, annotations, CATEGORIES)
    _touch_image(tmp_path, "clip_001.jpg")

    result = scan_flir_dataset(tmp_path)

    assert result.records == ()
    assert result.issues[0].category == ScanErrorCategory.INVALID_BBOX


def test_scan_rejects_entire_sample_instead_of_writing_partial_labels(tmp_path):
    images = [{"id": 1, "file_name": "clip_001.jpg", "width": 100, "height": 100}]
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [90, 90, 50, 50]},
    ]
    _write_annotations(tmp_path, images, annotations, CATEGORIES)
    _touch_image(tmp_path, "clip_001.jpg")

    result = scan_flir_dataset(tmp_path)

    assert result.records == ()
    assert [issue.category for issue in result.issues] == [ScanErrorCategory.INVALID_BBOX]


@pytest.mark.parametrize(
    "bbox",
    ([1, 2, 3], [1, 2, "not-a-number", 4], [1, 2, float("nan"), 4], [1, 2, 3, float("inf")]),
)
def test_scan_rejects_malformed_or_nonfinite_bbox_without_crashing(tmp_path, bbox):
    images = [{"id": 1, "file_name": "clip_001.jpg", "width": 100, "height": 100}]
    annotations = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": bbox}]
    _write_annotations(tmp_path, images, annotations, CATEGORIES)
    _touch_image(tmp_path, "clip_001.jpg")

    result = scan_flir_dataset(tmp_path)

    assert result.records == ()
    assert result.issues[0].category == ScanErrorCategory.INVALID_BBOX


def test_scan_ignores_unsupported_category_without_remapping(tmp_path):
    images = [{"id": 1, "file_name": "clip_001.jpg", "width": 100, "height": 100}]
    annotations = [{"id": 1, "image_id": 1, "category_id": 4, "bbox": [1, 1, 5, 5]}]
    _write_annotations(tmp_path, images, annotations, CATEGORIES)
    _touch_image(tmp_path, "clip_001.jpg")

    result = scan_flir_dataset(tmp_path)

    assert result.records[0].annotations == ()
    assert result.issues == ()


def test_scan_ignores_invalid_bbox_for_unsupported_category(tmp_path):
    images = [{"id": 1, "file_name": "clip_001.jpg", "width": 100, "height": 100}]
    annotations = [{"id": 1, "image_id": 1, "category_id": 4, "bbox": [90, 90, 50, 50]}]
    _write_annotations(tmp_path, images, annotations, CATEGORIES)
    _touch_image(tmp_path, "clip_001.jpg")

    result = scan_flir_dataset(tmp_path)

    assert len(result.records) == 1
    assert result.records[0].annotations == ()
    assert result.issues == ()


@pytest.mark.parametrize("path_kind", ["absolute", "parent"])
def test_scan_rejects_image_path_outside_source_root(tmp_path, path_kind):
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    file_name = str(outside) if path_kind == "absolute" else "../outside.jpg"
    images = [{"id": 1, "file_name": file_name, "width": 100, "height": 100}]
    _write_annotations(root, images, [], CATEGORIES)

    result = scan_flir_dataset(root)

    assert result.records == ()
    assert result.issues[0].category == ScanErrorCategory.INVALID_SOURCE_PATH


def test_scan_rejects_image_symlink_that_resolves_outside_root(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    (root / "linked.jpg").symlink_to(outside)
    images = [{"id": 1, "file_name": "linked.jpg", "width": 100, "height": 100}]
    _write_annotations(root, images, [], CATEGORIES)

    result = scan_flir_dataset(root)

    assert result.records == ()
    assert result.issues[0].category == ScanErrorCategory.INVALID_SOURCE_PATH


def test_scan_accepts_image_symlink_that_resolves_inside_root(tmp_path):
    (tmp_path / "target.jpg").write_bytes(b"inside")
    (tmp_path / "linked.jpg").symlink_to("target.jpg")
    images = [{"id": 1, "file_name": "linked.jpg", "width": 100, "height": 100}]
    _write_annotations(tmp_path, images, [], CATEGORIES)

    result = scan_flir_dataset(tmp_path)

    assert len(result.records) == 1
    assert result.issues == ()


@pytest.mark.parametrize("path_kind", ["absolute", "parent"])
def test_scan_rejects_annotation_path_outside_source_root(tmp_path, path_kind):
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.json"
    _write_annotations(tmp_path, [], [], CATEGORIES)
    (tmp_path / "thermal_annotations.json").rename(outside)
    annotation_filename = str(outside) if path_kind == "absolute" else "../outside.json"

    result = scan_flir_dataset(root, annotation_filename=annotation_filename)

    assert result.records == ()
    assert result.issues[0].category == ScanErrorCategory.INVALID_SOURCE_PATH


def test_scan_rejects_annotation_symlink_that_resolves_outside_root(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"images": [], "annotations": [], "categories": CATEGORIES}))
    (root / "linked.json").symlink_to(outside)

    result = scan_flir_dataset(root, annotation_filename="linked.json")

    assert result.records == ()
    assert result.issues[0].category == ScanErrorCategory.INVALID_SOURCE_PATH


def test_scan_accepts_dataset_root_supplied_through_symlink(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    _write_annotations(root, [], [], CATEGORIES)
    linked_root = tmp_path / "linked-dataset"
    linked_root.symlink_to(root, target_is_directory=True)

    result = scan_flir_dataset(linked_root)

    assert result.records == ()
    assert result.issues == ()


@pytest.mark.parametrize(
    "file_name,expected_key",
    [
        ("clip_001.jpg", "clip"),
        ("FLIR_video_00042_00007.jpg", "FLIR_video_00042"),
        ("no-trailing-digits.jpg", "no-trailing-digits"),
    ],
)
def test_sequence_key_groups_adjacent_frames(tmp_path, file_name, expected_key):
    from spectratwin.real_data.adapter import _sequence_key

    assert _sequence_key(file_name, extra_info=None) == expected_key


def test_sequence_key_prefers_source_backed_video_id():
    from spectratwin.real_data.adapter import _sequence_key

    assert (
        _sequence_key("frame-000108.jpg", {"video_id": "GzdKTLbkG5F7gAunM"}) == "GzdKTLbkG5F7gAunM"
    )
